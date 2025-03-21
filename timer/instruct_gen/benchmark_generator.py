import json
import time
import asyncio
import argparse
from pathlib import Path
from typing import Dict, Optional, List
from tqdm import tqdm
import vertexai
from vertexai.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models

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

def save_single_response(resp_data: Dict, output_file: Path, person_id: str):
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(resp_data, f, indent=2)
        print(f"Saved response for person_id: {person_id}")
    except Exception as e:
        print(f"Error saving response for person_id: {person_id}")
        print(f"Error details: {str(e)}")

def save_responses(responses: List[Dict], output_folder: Path):
    """
    Save responses in the same format as async_ehr_prompting.py
    """
    for response in responses:
        person_id = response["person_id"]
        output_file = output_folder / f"{person_id}.json"
        
        metadata = {
            "person_id": person_id,
            "window_token_count": response["window_token_count"],
            "window_percent_full": response["window_percent_full"],
            "start_date": response["start_date"],
            "end_date": response["end_date"],
            "response_data": response["response"]
        }
        
        save_single_response(metadata, output_file, person_id)
    
    print(f"Saved {len(responses)} responses to {output_folder}")

async def process_single_timeline_for_eval(timeline: Dict, args, model, semaphore) -> Optional[Dict]:
    try:
        processed_timeline = utils.process_ehr_for_temporal_eval(timeline, args.ehr_context)
        if not processed_timeline:
            return None
            
        prompt = utils.create_prompt_from_timeline(
            processed_timeline, 
            args.prompt_template
        )
        
        async with semaphore:
            response = await generate_async(prompt, model)
            
        return {
            "person_id": timeline.get('person_id', 'unknown'),
            "window_token_count": processed_timeline["window_token_count"],
            "window_percent_full": processed_timeline["window_percent_full"],
            "start_date": processed_timeline["start_date"],
            "end_date": processed_timeline["end_date"],
            "response": response
        }
    except Exception as e:
        print(f"Error processing timeline: {str(e)}")
        return None

async def main():
    parser = argparse.ArgumentParser(description="Generate temporal evaluation dataset from EHR records")
    parser.add_argument("--materialized_ehr_folder", type=str, required=True)
    parser.add_argument("--prompt_template", type=str, required=True)
    parser.add_argument("--ehr_context", type=int, default=16384)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--project_id", type=str)
    parser.add_argument("--location", type=str, default="us-central1")
    parser.add_argument("--model_name", type=str, default="gemini-1.5-pro-001")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--max_concurrent_calls", type=int, default=10)
    args = parser.parse_args()

    start_time = time.time()

    vertexai.init(project=args.project_id, location=args.location)
    model = GenerativeModel(args.model_name)
    semaphore = asyncio.Semaphore(args.max_concurrent_calls)

    # Load and filter timelines
    timelines = []
    for file in Path(args.materialized_ehr_folder).glob("*.jsonl"):
        with open(file) as f:
            for line in f:
                timelines.append(json.loads(line))
                if len(timelines) >= args.max_samples:
                    break
        if len(timelines) >= args.max_samples:
            break

    print(f"Processing {len(timelines)} timelines...")
    results = []
    async_tasks = []
    for timeline in timelines:
        task = process_single_timeline_for_eval(timeline, args, model, semaphore)
        async_tasks.append(task)

    for result in tqdm(
        await asyncio.gather(*async_tasks), 
        total=len(async_tasks),
        desc="Processing timelines"
    ):
        if result:
            results.append(result)

    # Save results
    output_path = Path(args.output_folder) / args.model_name / "temporal_eval" / str(args.ehr_context)
    save_responses(results, output_path)

    total_time = time.time() - start_time
    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"Average time per sample: {total_time / len(results):.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
