
import argparse
import datetime
import logging
import os
import time
from typing import Dict, List

import pandas as pd
import torch
import vllm
from vllm.lora.request import LoRARequest
import requests

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference using various LLM models"
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Path to HuggingFace model, 'gpt-4'/'gpt-4-32k' for OpenAI models, 'medlm-medium'/'medlm-large' for MedLM, or 'claude-3.5-sonnet' for Claude",
    )
    parser.add_argument(
        "--enable_lora",
        action="store_true",
        help="Enable LoRA for VLLM models",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        help="Path to adapted LoRA model",
    )
    parser.add_argument(
        "--generation_length",
        type=int,
        default=256,
        help="Target generation length in terms of tokens",
    )
    parser.add_argument(
        "--path_to_prompts",
        required=True,
        type=str,
        help="Path to pre-constructed LLM prompts",
    )
    parser.add_argument(
        "--path_to_save",
        required=True,
        type=str,
        help="Path where model responses should be saved",
    )
    parser.add_argument(
        '--context_length',
        type=int,
        default=16384,
        help='The maximum length of the context window for the model'
    )
    return parser.parse_args()

def get_compatible_dtype():
    """Check if GPUs are bfloat16 compatible"""
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        if hasattr(props, "name"):
            if "a100" in props.name.lower() or props.major >= 8:
                return torch.bfloat16
    logger.warning(
        "Bfloat16 is only supported on GPUs with compute capability of "
        "at least 8.0 (e.g., A100, H100). Dropping to float16."
    )
    return torch.float16

def inference_vllm(
    prompt_map: Dict[int, str],
    model_name: str,
    context_length: int,
    enable_lora: bool = False,
    lora_path: str = "",
    temperature: float = 0.4,
    top_p: float = 0.9,
    generation_length: int = 256,
) -> str:
    """Run inference using VLLM"""
    output_map = {}
    inst_ids = list(prompt_map.keys())
    instructions = [prompt_map[inst_id] for inst_id in inst_ids]
    print(f"Generating response for {len(instructions)} instructions")

    # GPU devices
    is_precision_16 = True
    torch_dtype = get_compatible_dtype() if is_precision_16 else torch.float32
    devices: List[int] = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    print(f"Visible CUDA devices: `{devices}`")
    print("Loading the vllm model...")

    sampling_params = vllm.SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=generation_length
    )
    os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
    
    if enable_lora:
        if not lora_path:
            raise ValueError("LoRA path must be provided if LoRA is enabled")
        llm = vllm.LLM(
            model_name,
            tensor_parallel_size=len(devices),
            dtype=torch_dtype,
            enable_lora=True,
            max_seq_len_to_capture=context_length,
            max_num_batched_tokens=context_length,
            max_model_len=context_length,
        )
        print("vllm model with LoRA successfully loaded...")
        outputs = llm.generate(instructions, sampling_params, lora_request=LoRARequest("adapter", 1, lora_path))
    else:
        llm = vllm.LLM(
            model_name,
            tensor_parallel_size=len(devices),
            dtype=torch_dtype,
            max_seq_len_to_capture=context_length,
            max_num_batched_tokens=context_length,
            max_model_len=context_length,
        )
        print("vllm model successfully loaded...")
        outputs = llm.generate(instructions, sampling_params)

    for inst_id, output in zip(inst_ids, outputs):
        generated_text = output.outputs[0].text
        output_map[inst_id] = generated_text
        print(f"Output for instruction {inst_id}: '{generated_text}'")

    return output_map

def inference_openai(
    prompt_map: Dict[int, str],
    model_name: str = "gpt-4",
    temperature: float = 0.4,
    top_p: float = 0.9,
    generation_length: int = 256,
) -> str:
    """Generate responses using OpenAI API with Azure endpoint configuration"""
    api_base = os.environ.get("OPENAI_API_BASE")
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_base or not api_key:
        raise ValueError(
            "Could not find API configuration. Please set OPENAI_API_BASE and "
            "OPENAI_API_KEY environment variables."
        )

    deployment_map = {
        "gpt-4o": "gpt-4o",
        "gpt-4-32k": "gpt-4-32k",
    }
    
    deployment_name = deployment_map.get(model_name.lower(), "gpt-4")
    api_version = "2023-05-15"

    def completion_with_backoff(messages, max_retries=30, initial_delay=10):
        """Helper function to handle API calls with exponential backoff"""
        for retry in range(max_retries):
            try:
                url = f"{api_base}/deployments/{deployment_name}/chat/completions?api-version={api_version}"
                headers = {
                    "Ocp-Apim-Subscription-Key": api_key,
                    "Content-Type": "application/json"
                }
                data = {
                    "messages": messages,
                    "max_tokens": generation_length,
                    "temperature": temperature,
                    "top_p": top_p
                }
                
                response = requests.post(url, headers=headers, json=data)
                
                # Check for rate limit errors specifically
                if response.status_code == 429:  # Too Many Requests
                    retry_after = int(response.headers.get('Retry-After', 60))
                    print(f"Rate limited. Waiting for {retry_after} seconds...")
                    time.sleep(retry_after)
                    continue
                
                response.raise_for_status()
                
                # Add base delay between successful requests
                time.sleep(3)  # Wait 3 seconds between successful requests
                
                return response.json()
                
            except requests.exceptions.RequestException as e:
                if retry == max_retries - 1:
                    print(f"Failed after {max_retries} attempts: {str(e)}")
                    return None
                    
                delay = min(initial_delay * (2 ** retry), 300)  # Cap at 5 minutes
                print(f"API call failed: {str(e)}. Retrying in {delay} seconds...")
                time.sleep(delay)
        
        return None

    output_map = {}
    total_requests = len(prompt_map)
    
    for idx, (inst_id, prompt) in enumerate(prompt_map.items(), 1):
        print(f"Processing request {idx}/{total_requests}")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        response = completion_with_backoff(messages)
        
        if response and "choices" in response:
            try:
                # First try the standard OpenAI API format
                if "message" in response["choices"][0]:
                    output_map[inst_id] = response["choices"][0]["message"]["content"]
                # Fallback to checking for direct content or text field
                elif "content" in response["choices"][0]:
                    output_map[inst_id] = response["choices"][0]["content"]
                elif "text" in response["choices"][0]:
                    output_map[inst_id] = response["choices"][0]["text"]
                else:
                    print(f"Unexpected response format for instruction {inst_id}. Response structure: {response['choices'][0]}")
                    output_map[inst_id] = "[Error: Unexpected response format]"
                print(f"Successfully processed instruction {inst_id}")
            except Exception as e:
                print(f"Error processing response for instruction {inst_id}: {str(e)}")
                output_map[inst_id] = "[Error: Failed to process response]"
        else:
            print(f"Failed to get response for instruction {inst_id}")
            output_map[inst_id] = "[Error: Failed to generate response]"
        
        # Add delay between requests to prevent rate limiting
        if idx % 10 == 0:  # Add longer pause every 10 requests
            pause_time = 30
            print(f"Taking a {pause_time}-second pause after {idx} requests...")
            time.sleep(pause_time)
        else:
            time.sleep(3)  # Base delay between requests

    return output_map

def inference_claude(
    prompt_map: Dict[int, str],
    temperature: float = 0.4,
    top_p: float = 0.9,
    generation_length: int = 256,
) -> str:
    """Generate responses using Claude 3.5 Sonnet v2 API"""
    api_base = os.environ.get("OPENAI_API_BASE")
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "Could not find API configuration. Please set OPENAI_API_KEY "
            "environment variable."
        )

    model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"

    def completion_with_backoff(prompt_text, max_retries=30, initial_delay=10):
        """Helper function to handle API calls with exponential backoff"""
        headers = {
            "Ocp-Apim-Subscription-Key": api_key,
            "Content-Type": "application/json"
        }
        
        for retry in range(max_retries):
            try:
                data = {
                    "model_id": model_id,
                    "prompt_text": prompt_text,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": generation_length
                }
                
                response = requests.post(api_base, headers=headers, json=data)
                
                # Check for rate limit errors
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    print(f"Rate limited. Waiting for {retry_after} seconds...")
                    time.sleep(retry_after)
                    continue
                
                response.raise_for_status()
                
                # Add base delay between successful requests
                time.sleep(3)
                
                return response.json()
                
            except requests.exceptions.RequestException as e:
                if retry == max_retries - 1:
                    print(f"Failed after {max_retries} attempts: {str(e)}")
                    return None
                    
                delay = min(initial_delay * (2 ** retry), 300)  # Cap at 5 minutes
                print(f"API call failed: {str(e)}. Retrying in {delay} seconds...")
                time.sleep(delay)
        
        return None

    output_map = {}
    total_requests = len(prompt_map)
    
    for idx, (inst_id, prompt) in enumerate(prompt_map.items(), 1):
        print(f"Processing request {idx}/{total_requests}")
        try:
            response = completion_with_backoff(prompt)
            
            if response and isinstance(response, dict):
                if "content" in response and isinstance(response["content"], list):
                    text_contents = []
                    for content_item in response["content"]:
                        if isinstance(content_item, dict) and "type" in content_item and "text" in content_item:
                            if content_item["type"] == "text":
                                text_contents.append(content_item["text"])
                    
                    if text_contents:
                        output_map[inst_id] = " ".join(text_contents)
                        print(f"Successfully processed instruction {inst_id}")
                        continue
                
                # Fallback handlers
                if "completion" in response:
                    output_map[inst_id] = response["completion"]
                elif "text" in response:
                    output_map[inst_id] = response["text"]
                else:
                    print(f"Unknown response format for instruction {inst_id}. Response keys: {list(response.keys())}")
                    output_map[inst_id] = "[Error: Unknown response format]"
            else:
                print(f"Invalid response format for instruction {inst_id}")
                output_map[inst_id] = "[Error: Invalid response format]"
                
        except Exception as e:
            print(f"Exception processing instruction {inst_id}: {str(e)}")
            output_map[inst_id] = f"[Error: {str(e)}]"
            
        # Add delay between requests to prevent rate limiting
        if idx % 10 == 0:  # Add longer pause every 10 requests
            pause_time = 30
            print(f"Taking a {pause_time}-second pause after {idx} requests...")
            time.sleep(pause_time)
        else:
            time.sleep(3)  # Base delay between requests

    return output_map

def inference_medpalm(
    prompt_map: Dict[int, str],
    model_name: str,
    context_length: int = 16384,  # Added context_length parameter
    temperature: float = 0.2,
    top_p: float = 0.8,
    do_sample: bool = True,
    generation_length: int = 256,
    project_id = None
) -> str:
    """
    Run inference using Google's MedLM models with context length control
    Args:
        prompt_map: Dictionary mapping instruction IDs to prompts
        model_name: Name of the MedLM model to use
        context_length: Maximum context length allowed
        temperature: Temperature for generation
        top_p: Top p for generation
        do_sample: Whether to use sampling
        generation_length: Maximum length of generated response
    Returns:
        Dictionary mapping instruction IDs to generated responses
    """
    # Imports specific for MedLM
    from google.cloud import aiplatform
    from google.cloud.aiplatform.gapic.schema import predict
    from google.protobuf import json_format
    from google.protobuf.struct_pb2 import Value
    import tiktoken

    def get_tokenizer():
        """Get tiktoken tokenizer"""
        return tiktoken.get_encoding("cl100k_base")

    def truncate_text(text: str, max_tokens: int, tokenizer) -> str:
        """Truncate text to fit within token limit"""
        encoded = tokenizer.encode(text)
        if len(encoded) > max_tokens:
            encoded = encoded[-max_tokens:]  # Keep the most recent tokens
        return tokenizer.decode(encoded)

    # Initialize tokenizer
    tokenizer = get_tokenizer()

    # Calculate available context length for input
    # Reserve tokens for model response and any special tokens
    available_context = context_length - generation_length - 100  # 100 tokens buffer for special tokens

    # Map model names to API endpoints
    endpoint_map = {
        "medlm-medium": f"projects/{project_id}/locations/us-central1/publishers/google/models/medlm-medium",
        "medlm-large": f"projects/{project_id}/locations/us-central1/publishers/google/models/medlm-large"
    }
    
    api_endpoint = endpoint_map.get(model_name.lower())
    if not api_endpoint:
        raise ValueError(f"Unknown MedLM model: {model_name}")

    output_map = {}
    inst_ids = list(prompt_map.keys())
    instructions = [prompt_map[inst_id] for inst_id in inst_ids]

    # Initialize client
    client_options = {"api_endpoint": "us-central1-aiplatform.googleapis.com"}
    print("Instantiating a prediction service client with Google API")
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

    for inst_id, instruction in zip(inst_ids, instructions):
        # Truncate instruction if needed
        truncated_instruction = truncate_text(instruction, available_context, tokenizer)
        
        # Create instance with truncated instruction
        instance_dict = {"content": truncated_instruction}
        instance = json_format.ParseDict(instance_dict, Value())
        instances = [instance]
        
        parameters_dict = {
            "candidateCount": 1,
            "maxOutputTokens": generation_length,
            "temperature": temperature,
            "topP": top_p,
            "topK": 40,
        }
        parameters = json_format.ParseDict(parameters_dict, Value())
        
        # Print token counts for monitoring
        original_tokens = len(tokenizer.encode(instruction))
        truncated_tokens = len(tokenizer.encode(truncated_instruction))
        if original_tokens != truncated_tokens:
            print(f"Truncated instruction {inst_id} from {original_tokens} to {truncated_tokens} tokens")
        
        response = client.predict(
            endpoint=api_endpoint,
            instances=instances,
            parameters=parameters
        )
        
        predictions = response.predictions
        for prediction in predictions:
            output_map[inst_id] = dict(prediction)["content"]
            print(f"instruction = {truncated_instruction}\nprediction = {output_map[inst_id]}")
            time.sleep(10)  # Rate limiting
            
    return output_map

def main():
    print("Starting inference...")
    args = parse_args()
    print(f"Model name: {args.model}")
    model_name = args.model
    
    # Load prompts
    packed_prompts_df = pd.read_csv(args.path_to_prompts)
    instructionid_to_prompt_map = (
        packed_prompts_df[["instruction_id", "prompt"]]
        .set_index("instruction_id")
        .to_dict()
        .get("prompt")
    )

    # Generate LLM responses
    print("Running prompts through the model...")
    if model_name.lower() in ["gpt-4o", "gpt-4-32k"]:
        outputs = inference_openai(
            instructionid_to_prompt_map,
            model_name,
            generation_length=args.generation_length,
        )
    elif model_name.lower() in ["medlm-medium", "medlm-large"]:
        outputs = inference_medpalm(
            instructionid_to_prompt_map,
            model_name.lower(),
            context_length=args.context_length,  # Add this line
            generation_length=args.generation_length,
        )
    elif model_name.lower() == "claude-3.5-sonnet":
        outputs = inference_claude(
            instructionid_to_prompt_map,
            generation_length=args.generation_length,
        )
    else:  # VLLM models
        outputs = inference_vllm(
            prompt_map=instructionid_to_prompt_map,
            model_name=model_name,
            context_length=args.context_length,
            enable_lora=args.enable_lora,
            lora_path=args.lora_path,
            generation_length=args.generation_length,
        )

    # Pack results into DataFrame and save
    output_df_rows = []
    for _, row in packed_prompts_df.iterrows():
        instruction_id = row["instruction_id"]
        patient_id = row["patient_id"]
        if instruction_id not in outputs:
            print(f"WARNING: Instruction ID {instruction_id} had no generated output")
            continue
            
        output_df_rows.append({
            "instruction_id": instruction_id,
            "patient_id": patient_id,
            "prompt": instructionid_to_prompt_map[instruction_id],
            "model_response": outputs[instruction_id],
            "model_name": model_name,
            "generation_length": args.generation_length,
        })
        
    pd.DataFrame(output_df_rows).to_csv(args.path_to_save, index=False)

if __name__ == "__main__":
    main()
