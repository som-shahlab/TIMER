import argparse
import json
from pathlib import Path
import pandas as pd
import random
from typing import Dict, List, Tuple
import tiktoken
import utils
import os
from xml.dom import minidom


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert synthetic instruction-response pairs into MedAlign format"
    )
    parser.add_argument(
        "--input_folder",
        required=True,
        type=str,
        help="Root folder containing model outputs",
    )
    parser.add_argument(
        "--materialized_ehr_folder", 
        required=True,
        type=str,
        help="Folder containing materialized EHR JSONL files"
    )
    parser.add_argument(
        "--output_inference_csv",
        required=True,
        type=str,
        help="Path to save CSV for inference.py",
    )
    parser.add_argument(
        "--output_reference_csv", 
        required=True,
        type=str,
        help="Path to save CSV for llm_as_judge.py reference answers",
    )
    parser.add_argument(
        "--output_xml_dir",
        required=True,
        type=str,
        help="Directory to save the XML files for each patient"
    )
    parser.add_argument(
        "--ehr_context",
        type=int,
        default=16384,
        help="Context length for EHR truncation"
    )
    parser.add_argument(
        "--total_instructions",
        type=int,
        required=True,
        help="Total number of instruction-response pairs to sample (e.g., 303)"
    )
    parser.add_argument(
        "--instructions_per_person",
        type=int,
        required=True,
        help="Number of instruction-response pairs to sample per person (e.g., 2 or 3)"
    )
    return parser.parse_args()

def export_ehr_to_xml(ehr_data: Dict, output_dir: str):
    """
    Export EHR text data to XML file
    
    Args:
        ehr_data: Dictionary containing EHR data with XML string in 'text' field
        output_dir: Directory to save XML files
    """
    try:
        person_id = str(ehr_data['person_id'])
        xml_str = ehr_data['text']
        
        # Parse and pretty print the XML
        dom = minidom.parseString(xml_str)
        # Get pretty XML string but skip the first line containing the XML declaration
        pretty_xml = '\n'.join(dom.toprettyxml(indent="  ").split('\n')[1:])
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save XML file
        output_path = os.path.join(output_dir, f"EHR_{person_id}.xml")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)
        
        print(f"Saved XML file for person {person_id} to {output_path}")
        
    except Exception as e:
        print(f"Error exporting XML for person {ehr_data.get('person_id', 'unknown')}: {e}")

def load_materialized_ehrs(folder_path: str) -> Dict[str, Dict]:
    """Load all materialized EHRs from JSONL files"""
    ehrs = {}
    print(f"Loading EHRs from {folder_path}")
    for file in Path(folder_path).glob("*.jsonl"):
        print(f"Processing JSONL file: {file}")
        with open(file) as f:
            for line in f:
                timeline = json.loads(line)
                person_id = str(timeline['person_id'])
                ehrs[person_id] = timeline
    print(f"Loaded {len(ehrs)} EHRs")
    return ehrs

def get_tokenizer():
    """Get tiktoken tokenizer as used in preprocessing.py"""
    return tiktoken.get_encoding("cl100k_base")

def truncate_ehr(ehr_text: str, context_length: int, generation_length: int, template_text: str, tokenizer) -> str:
    """
    Truncate EHR text to fit within context length, accounting for generation length and template.
    Following preprocessing.py logic.
    """
    # Calculate token counts
    template_tokens = len(tokenizer.encode(template_text))
    target_ehr_length = context_length - generation_length - template_tokens
    
    # Do initial truncation with fast tokenizer
    fast_encoded = tokenizer.encode(ehr_text)
    fast_encoded_truncated = fast_encoded[-target_ehr_length:]
    truncated_ehr = tokenizer.decode(fast_encoded_truncated)
    
    return truncated_ehr

def validate_instruction_response(inst_response):
    """Validate the instruction-response format"""
    try:
        if isinstance(inst_response, dict):
            if 'instruction' not in inst_response or 'response' not in inst_response:
                print(f"Warning: Missing required fields in instruction-response: {inst_response}")
                return None
            return inst_response
        elif isinstance(inst_response, str):
            print(f"Warning: Found string instead of dict for instruction-response: {inst_response}")
            return None
        else:
            print(f"Warning: Invalid type for instruction-response: {type(inst_response)}")
            return None
    except Exception as e:
        print(f"Error validating instruction-response: {e}")
        return None

def validate_response_data(response_data):
    """Validate the response_data array"""
    if not isinstance(response_data, list):
        print(f"Warning: response_data is not a list: {type(response_data)}")
        return []
    
    valid_responses = []
    for inst_response in response_data:
        validated = validate_instruction_response(inst_response)
        if validated:
            valid_responses.append(validated)
    return valid_responses

def load_generated_evaluation(input_folder: str) -> List[Dict]:
    """Load generated instruction-response pairs from nested directory structure"""
    data = []
    eval_dir = Path(input_folder)
    if not eval_dir.exists():
        raise ValueError(f"Directory not found: {eval_dir}")
    
    print(f"Loading evaluation data from {eval_dir}")
    for json_file in eval_dir.glob("*.json"):
        print(f"Processing file: {json_file}")
        with open(json_file, 'r') as f:
            try:
                json_data = json.load(f)
                if all(k in json_data for k in ['person_id', 'response_data']):
                    json_data['person_id'] = str(json_data['person_id'])
                    json_data['filename'] = f"EHR_{json_data['person_id']}.xml"
                    data.append(json_data)
                    print(f"Loaded evaluation data with {len(json_data['response_data'])} instructions for person_id: {json_data['person_id']}")
            except json.JSONDecodeError as e:
                print(f"Error reading {json_file}: {e}")
                
    print(f"Loaded {len(data)} evaluation files")
    return data

def create_instruction_ids(eval_data: List[Dict]) -> Dict[Tuple[str, int], int]:
    """
    Create consistent instruction IDs for each person_id and instruction index combination.
    
    Returns:
        Dictionary mapping (person_id, instruction_index) to instruction_id
    """
    instruction_ids = {}
    for item in eval_data:
        person_id = str(item.get('person_id', ''))
        if not person_id:
            continue
            
        inst_id_base = random.randint(1000000, 8999999)
        response_data = validate_response_data(item.get('response_data', []))
        
        for i in range(len(response_data)):
            instruction_ids[(person_id, i)] = inst_id_base + i
            
    return instruction_ids

def create_inference_csv(eval_data: List[Dict], ehrs: Dict[str, Dict], context_size: int, output_path: str, instruction_ids: Dict[Tuple[str, int], int]):
    """Create CSV file for inference.py input with truncated EHRs"""
    rows = []
    print("Creating inference CSV rows...")
    
    tokenizer = get_tokenizer()
    prompt_template = """
        You are an expert medical doctor. Based on the provided electronic health record (EHR), please respond to the following question.

        [EHR]
        {ehr}

        [Question]
        {question}

        [Output Format]
        Please provide your response in a clear and concise manner, focusing on the relevant information from the EHR that supports your answer.
    """
    
    for item in eval_data:
        try:
            person_id = str(item.get('person_id', ''))
            if not person_id:
                print("Warning: Missing person_id in evaluation data")
                continue
                
            print(f"Processing person_id: {person_id}")
            
            if person_id not in ehrs:
                print(f"Warning: No EHR found for person_id {person_id}")
                continue
            
            response_data = validate_response_data(item.get('response_data', []))
            if not response_data:
                print(f"Warning: No valid instructions found for person_id {person_id}")
                continue
                
            print(f"Processing {len(response_data)} instructions for person_id {person_id}")
            
            try:
                processed_timeline = utils.process_ehr_for_temporal_eval(
                    ehrs[person_id], 
                    context_size
                )
                if not processed_timeline:
                    print(f"Warning: Could not process EHR for person_id {person_id}")
                    continue
                
                recent_window_text = processed_timeline['text']
                
                try:
                    truncated_ehr = truncate_ehr(
                        ehr_text=recent_window_text,
                        context_length=context_size,
                        generation_length=256,
                        template_text=prompt_template,
                        tokenizer=tokenizer
                    )
                except Exception as e:
                    print(f"Error truncating EHR for person_id {person_id}: {e}")
                    continue
                
                for i, inst_response in enumerate(response_data):
                    try:
                        instruction_id = instruction_ids.get((person_id, i))
                        if instruction_id is None:
                            print(f"Warning: No instruction ID found for {person_id}, index {i}")
                            continue
                            
                        instruction = inst_response.get('instruction', '')
                        if not instruction:
                            print(f"Warning: Empty instruction for {person_id}, index {i}")
                            continue
                            
                        try:
                            full_prompt = prompt_template.format(ehr=truncated_ehr, question=instruction)
                        except Exception as e:
                            print(f"Error creating prompt for {person_id}, instruction {i}: {e}")
                            continue
                            
                        rows.append({
                            'instruction_id': instruction_id,
                            'patient_id': person_id,
                            'instruction': instruction,
                            'ehr': truncated_ehr,
                            'prompt': full_prompt,
                            'prompt_template': 'temporal_eval',
                            'context_length': context_size,
                            'generation_length': 256
                        })
                        print(f"Added instruction {instruction_id} for person_id {person_id}")
                    except Exception as e:
                        print(f"Error processing instruction {i} for person_id {person_id}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error processing timeline for person_id {person_id}: {e}")
                continue
                
        except Exception as e:
            print(f"Error processing evaluation data item: {e}")
            continue
    
    print(f"Created {len(rows)} total rows for inference CSV")
    if rows:
        try:
            df = pd.DataFrame(rows)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"Saved inference input CSV with {len(rows)} rows to {output_path}")
        except Exception as e:
            print(f"Error saving CSV file: {e}")
    else:
        print("Warning: No rows created for inference CSV")

def create_reference_csv(eval_data: List[Dict], output_path: str, instruction_ids: Dict[Tuple[str, int], int]):
    """Create CSV file for llm_as_judge.py reference answers"""
    rows = []
    print("Creating reference CSV rows...")
    
    for item in eval_data:
        try:
            person_id = str(item.get('person_id', ''))
            if not person_id:
                print("Warning: Missing person_id in evaluation data")
                continue
                
            response_data = validate_response_data(item.get('response_data', []))
            if not response_data:
                print(f"Warning: No valid instructions found for person_id {person_id}")
                continue
                
            print(f"Processing {len(response_data)} responses for person_id {person_id}")
            
            for i, inst_response in enumerate(response_data):
                try:
                    instruction_id = instruction_ids.get((person_id, i))
                    if instruction_id is None:
                        print(f"Warning: No instruction ID found for {person_id}, index {i}")
                        continue
                        
                    instruction = inst_response.get('instruction', '')
                    response = inst_response.get('response', '')
                    
                    if not instruction or not response:
                        print(f"Warning: Missing instruction or response for {person_id}, index {i}")
                        continue
                        
                    rows.append({
                        'instruction_id': instruction_id,
                        'filename': f"EHR_{person_id}.xml",
                        'question': instruction,
                        'is_applicable': 'yes',
                        'is_sufficient': 'yes',
                        'required_characteristics': '',
                        'clinician_response': response,
                        'evidence': inst_response.get('evidence', ''),
                    })
                except Exception as e:
                    print(f"Error processing instruction {i} for person_id {person_id}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error processing evaluation data item: {e}")
            continue
    
    print(f"Created {len(rows)} total rows for reference CSV")
    if rows:
        try:
            df = pd.DataFrame(rows)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"Saved reference CSV with {len(rows)} rows to {output_path}")
        except Exception as e:
            print(f"Error saving CSV file: {e}")
    else:
        print("Warning: No rows created for reference CSV")

def sample_exact_instructions(eval_data: List[Dict], total_instructions: int, instructions_per_person: int) -> List[Dict]:
    """
    Sample exact number of instructions with fixed number per person.
    
    Args:
        eval_data: List of patient data with instructions
        total_instructions: Total number of instruction-response pairs desired (e.g., 303)
        instructions_per_person: Number of pairs to sample per person (e.g., 2 or 3)
    
    Returns:
        List of sampled patient data with reduced instruction sets
    """
    print(f"\nPerforming exact sampling:")
    print(f"Target total instructions: {total_instructions}")
    print(f"Instructions per person: {instructions_per_person}")
    
    needed_patients = total_instructions // instructions_per_person
    if total_instructions % instructions_per_person != 0:
        print(f"Warning: {total_instructions} is not divisible by {instructions_per_person}")
        print(f"Will sample {needed_patients * instructions_per_person} instructions instead")
    
    print(f"Need {needed_patients} patients with {instructions_per_person} instructions each")
    
    # Filter for eligible patients (those with enough instructions)
    eligible_patients = [
        item for item in eval_data 
        if len(validate_response_data(item.get('response_data', []))) >= instructions_per_person
    ]
    print(f"Found {len(eligible_patients)} eligible patients with >= {instructions_per_person} instructions")
    
    if len(eligible_patients) < needed_patients:
        raise ValueError(
            f"Not enough eligible patients! Need {needed_patients} but only found {len(eligible_patients)} "
            f"with {instructions_per_person} or more instructions."
        )

    import random
    sampled_patients = random.sample(eligible_patients, needed_patients)
    
    sampled_data = []
    total_sampled = 0
    
    for item in sampled_patients:
        person_id = str(item.get('person_id', ''))
        response_data = validate_response_data(item.get('response_data', []))
        
        sampled_responses = random.sample(response_data, instructions_per_person)
        
        sampled_item = {
            'person_id': person_id,
            'filename': item.get('filename'),
            'window_token_count': item.get('window_token_count'),
            'window_percent_full': item.get('window_percent_full'),
            'start_date': item.get('start_date'),
            'end_date': item.get('end_date'),
            'response_data': sampled_responses
        }
        sampled_data.append(sampled_item)
        total_sampled += len(sampled_responses)
        
        print(f"Sampled {instructions_per_person} instructions for person {person_id}")
    
    print(f"\nSampling complete:")
    print(f"Total sampled instructions: {total_sampled}")
    print(f"Total sampled patients: {len(sampled_data)}")
    print(f"Instructions per patient: {instructions_per_person}")
    return sampled_data


def main():
    args = parse_args()
    
    if os.path.exists(args.output_reference_csv):
        print(f"\nFound existing reference CSV at {args.output_reference_csv}")
        print("Will only export XML files for existing person_ids...")
        
        df = pd.read_csv(args.output_reference_csv)
        person_ids = df['person_id'].astype(str).unique()
        print(f"Found {len(person_ids)} unique person IDs in reference answers")
        
        print("\nLoading materialized EHRs...")
        ehrs = load_materialized_ehrs(args.materialized_ehr_folder)
        
        print(f"\nExporting EHRs to XML format to {args.output_xml_dir}...")
        exported_count = 0
        missing_count = 0
        
        for person_id in person_ids:
            if person_id in ehrs:
                export_ehr_to_xml(ehrs[person_id], args.output_xml_dir)
                exported_count += 1
            else:
                print(f"Warning: No EHR found for person_id {person_id}")
                missing_count += 1
        
        print(f"\nExport complete:")
        print(f"Total person IDs from reference answers: {len(person_ids)}")
        print(f"Successfully exported: {exported_count}")
        print(f"Missing EHRs: {missing_count}")
        
    else:
        print("\nNo existing reference CSV found. Running full workflow...")
        
        print("\nStep 1: Loading materialized EHRs...")
        ehrs = load_materialized_ehrs(args.materialized_ehr_folder)
        
        print("\nStep 2: Loading generated evaluation data...")
        eval_data = load_generated_evaluation(args.input_folder)
        
        if not eval_data:
            print("Error: No evaluation data loaded!")
            return
            
        sampled_data = sample_exact_instructions(
            eval_data, 
            args.total_instructions,
            args.instructions_per_person
        )
        
        print(f"\nStep 2.5: Exporting EHRs to XML format to {args.output_xml_dir}...")
        sampled_person_ids = {str(item['person_id']) for item in sampled_data}
        exported_count = 0
        missing_count = 0
        
        for person_id in sampled_person_ids:
            if person_id in ehrs:
                export_ehr_to_xml(ehrs[person_id], args.output_xml_dir)
                exported_count += 1
            else:
                print(f"Warning: No EHR found for person_id {person_id}")
                missing_count += 1
        
        print(f"\nExport complete:")
        print(f"Total sampled person IDs: {len(sampled_person_ids)}")
        print(f"Successfully exported: {exported_count}")
        print(f"Missing EHRs: {missing_count}")
        
        instruction_ids = create_instruction_ids(sampled_data)
        
        print("\nStep 3: Creating inference input CSV...")
        create_inference_csv(sampled_data, ehrs, args.ehr_context, args.output_inference_csv, instruction_ids)
        
        print("\nStep 4: Creating reference answers CSV...")
        create_reference_csv(sampled_data, args.output_reference_csv, instruction_ids)

if __name__ == "__main__":
    main()
