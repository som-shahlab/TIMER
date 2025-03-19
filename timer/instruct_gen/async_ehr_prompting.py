import json
import time
import asyncio
import glob
import argparse
from argparse import ArgumentTypeError
from pathlib import Path
from tqdm import tqdm
from google.cloud import bigquery
import vertexai
from vertexai.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models
from typing import Dict, Optional

import utils

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

def merge_jsonl_files(input_folder, output_file):
    seen_records = set()
    with open(output_file, 'w') as outfile:
        for filename in glob.glob(f"{input_folder}/*.jsonl"):
            with open(filename, 'r') as infile:
                for line in infile:
                    record = json.loads(line)
                    unique_id = str(record.get('person_id', ''))
                    if unique_id not in seen_records:
                        outfile.write(line)
                        seen_records.add(unique_id)
    print(f"Merged {len(seen_records)} unique records into {output_file}")

def get_context_path_safe(ehr_context):
    """Convert ehr_context value to a path-safe string."""
    if isinstance(ehr_context, int) or (isinstance(ehr_context, str) and ehr_context.isdigit()):
        return f"context_{ehr_context}"
    return str(ehr_context)

def validate_ehr_context(value):
    """
    Custom validation for ehr_context argument.
    Accepts either predefined strings or positive integers.
    Returns:
    - String values unchanged ("full", "last_five")
    - Integer values as integers
    - For path construction, numeric values are converted to "context_{value}" format
    """
    if value in ["full", "last_five"]:
        return value
    try:
        context_length = int(value)
        if context_length <= 0:
            raise ArgumentTypeError(f"Context length must be positive, got {value}")
        return context_length
    except ValueError:
        raise ArgumentTypeError(
            f"Invalid ehr_context value: {value}. Must be either 'full', 'last_five'"
            "or a positive integer for context length."
        )
    

async def generate_async(prompt, model, max_retries=3, initial_delay=1):
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                contents=prompt,
                generation_config={"response_mime_type": "application/json"},
                stream=False
            )
            try:
                json_response = json.loads(response.text)
                return json_response  
            except json.JSONDecodeError:
                print(f"Generated content is not valid JSON. Retrying...")
                continue  
                
        except Exception as e:
            error_message = str(e)
            if "500" in error_message:  
                if attempt < max_retries - 1:
                    delay = initial_delay * (2 ** attempt)  # Exponential backoff
                    print(f"Encountered 500 error. Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    print(f"Max retries reached. Unable to generate content.")
                    return {"error": f"Max retries reached - {error_message}"}
            else:
                print(f"Error in generate_async: {error_message}")
                return {"error": error_message}

    return {"error": "Failed to generate valid JSON content after multiple attempts"}


async def process_single_timeline(timeline, args, model, semaphore, windowed_jsonl_file):
    person_id = timeline.get('person_id', 'unknown')
    print(f"Processing timeline for person_id: {person_id}")
    
    try:   
        context_windows = utils.process_ehr_context(
            timeline, 
            args.ehr_context, 
            windowed_jsonl_file
        )
        responses = []
        
        for window in context_windows:
            if args.prompt_method == "general":
                prompt = utils.create_prompt_from_timeline(
                    window, 
                    args.prompt_template
                )
                async with semaphore:
                    response = await generate_async(prompt, model)
                responses.append({
                    "window_index": window.get("window_index", 0),
                    "response": response,
                    "window_token_count": window.get("window_token_count"),
                    "window_percent_full": window.get("window_percent_full"),
                    "start_date": window.get("start_date"),
                    "end_date": window.get("end_date")
                })
                specialty = None
                visit_occurrence_id = None
            elif args.prompt_method == "persona":
                prompt, specialty, visit_occurrence_id = utils.create_persona_prompt_from_timeline(
                    window, 
                    args.prompt_template,
                    person_id
                )
                async with semaphore:
                    response = await generate_async(prompt, model)
                responses.append({
                    "window_index": window.get("window_index", 0),
                    "response": response,
                    "window_token_count": window.get("window_token_count"),
                    "window_percent_full": window.get("window_percent_full"),
                    "start_date": window.get("start_date"),
                    "end_date": window.get("end_date")
                })
            else:
                raise ValueError(f"Unknown prompt method: {args.prompt_method}")
        
        print(f"Completed processing for person_id: {person_id} with {len(responses)} context windows")
        return {
            "person_id": person_id,
            "visit_occurrence_id": visit_occurrence_id,
            "responses": responses,
            "specialty": specialty
        }
    except Exception as e:
        print(f"Error processing timeline for person_id {person_id}: {str(e)}")
        return None

async def process_timelines(merged_jsonl_file, args, windowed_jsonl_file):
    vertexai.init(project=args.project_id, location=args.location)
    model = GenerativeModel(args.model_name)
    semaphore = asyncio.Semaphore(args.max_concurrent_calls)

    with open(merged_jsonl_file, 'r') as f:
        timelines = [json.loads(line) for line in f]
    total_timelines = min(len(timelines), args.max_samples) if args.max_samples else len(timelines)
    print(f"Found {total_timelines} total timelines")

    results = []
    processed_count = 0
    for timeline in tqdm(timelines[:total_timelines], total=total_timelines, desc="Processing timelines"):
        try:
            result = await process_single_timeline(timeline, args, model, semaphore, windowed_jsonl_file)
            if result is not None:
                results.append(result)
            processed_count += 1
            if processed_count % 10 == 0 or processed_count == total_timelines:
                print(f"Processed {processed_count}/{total_timelines} samples")
        except Exception as e:
            print(f"Error processing timeline: {e}")

    return results

def save_single_response(resp_data: Dict, output_file: Path, person_id: str, template_name: Optional[str] = None):
    try:
        with open(output_file, 'w') as f:
            json.dump(resp_data, f, indent=2)
        print(f"Saved response for person_id: {person_id}" + 
              (f", template: {template_name}" if template_name else ""))
    except Exception as e:
        print(f"Error saving response for person_id: {person_id}" + 
              (f", template: {template_name}" if template_name else ""))
        print(f"Error details: {str(e)}")

def save_responses(responses, output_folder, prompt_method):
    """
    Save responses with window metadata including date ranges.
    """
    for response in responses:
        person_id = response["person_id"]
        
        for window_response in response["responses"]:
            if prompt_method == "general":
                output_file = Path(output_folder) / f"{person_id}_{window_response['window_index']}.json"
            elif prompt_method == "persona":
                specialty = response.get("specialty", "unknown")
                output_file = Path(output_folder) / f"{person_id}_{specialty}_{window_response['window_index']}.json"
            
            metadata = {
                "person_id": person_id,
                "window_index": window_response["window_index"],
                "window_token_count": window_response["window_token_count"],
                "window_percent_full": window_response["window_percent_full"],
                "start_date": window_response["start_date"],
                "end_date": window_response["end_date"],
                "response_data": window_response["response"]
            }
                
            output_file.parent.mkdir(parents=True, exist_ok=True)
            save_single_response(metadata, output_file, person_id)
            
            print(f"Saved response for person_id: {person_id}, "
                  f"window: {window_response['window_index']}, "
                  f"dates: {window_response['start_date']} to {window_response['end_date']}")
    
    print(f"Saved responses from {len(responses)} timelines to {output_folder}")


async def main(args):
    start_time = time.time()
    
    merged_jsonl_file = Path(args.materialized_ehr_folder) / f"merged_timelines.jsonl"
    windowed_dir = Path(args.materialized_ehr_folder) / "windowed"
    windowed_dir.mkdir(parents=True, exist_ok=True)
    windowed_jsonl_file = windowed_dir / f"windowed_timelines_{args.prompt_method}_{args.ehr_context}.jsonl"
    
    merge_jsonl_files(args.materialized_ehr_folder, merged_jsonl_file)
    
    print("Processing timelines...")
    responses = await process_timelines(merged_jsonl_file, args, windowed_jsonl_file)
    print(f"Processed {len(responses)} samples")

    if args.max_samples:
        print(f"Limited to {args.max_samples} samples for testing")

    if responses:
        context_path = get_context_path_safe(args.ehr_context)
        output_folder = Path(args.output_folder) / args.model_name / args.prompt_method / context_path / Path(args.prompt_template).stem
        save_responses(responses, output_folder, args.prompt_method)
        
        total_time = time.time() - start_time
        print(f"Total time taken: {total_time:.2f} seconds")
        print(f"Average time per sample: {total_time / len(responses):.2f} seconds")
    else:
        print("No responses were generated. Check the logs for more information.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic instruction-response pairs from materialized patient timelines using async processing.")
    parser.add_argument("--materialized_ehr_folder", type=str, help="Folder containing materialized EHR JSONL files")
    parser.add_argument("--prompt_template", type=str)
    parser.add_argument("--prompt_method", type=str, default="persona", help="Prompt method: general, persona")
    parser.add_argument("--ehr_context",type=validate_ehr_context,default=16384,help="EHR context to use: 'full', 'last_five', or a positive integer for context length (e.g., 8192)")
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--project_id", type=str)
    parser.add_argument("--location", type=str, default="us-central1", help="Google Cloud location")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--max_concurrent_calls")
    parser.add_argument("--csv_file")
    parser.add_argument("--dataset_id")
    parser.add_argument("--table_id", type=str)
    args = parser.parse_args()

    asyncio.run(main(args))
