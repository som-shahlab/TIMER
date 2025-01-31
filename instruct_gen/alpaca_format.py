import os
import json
import re
import random
import argparse
from itertools import islice
import re
from transformers import AutoTokenizer
import tiktoken
from tqdm import tqdm

def parse_instruction_response_pairs(file_content):
    file_content = re.sub(r'SAFETY_HAZARD.*?(?=\n\n|\Z)', '', file_content, flags=re.DOTALL)
    
    file_content = file_content.replace('\r\n', '\n').replace('\r', '\n')
    
    patterns = [
        r'(?:^|\n)(?:[Pp]air \d+[:.]?\s*)?(?:\d+\.?\s*)?[Ii]nstruction:?\s*(.*?)\n+\s*[Rr]esponse:?\s*(.*?)(?=\n(?:[Pp]air \d+[:.]?\s*)?(?:\d+\.?\s*)?[Ii]nstruction|$)',
        r'(?:^|\n)(?:[Pp]air \d+[:.]?\s*)?(?:\*\*)?[Ii]nstruction ?\d*:?(?:\*\*)?\s*(.*?)\n+\s*(?:\*\*)?[Rr]esponse(?: \d*)?:?(?:\*\*)?\s*(.*?)(?=\n(?:[Pp]air \d+[:.]?\s*)?(?:\*\*)?[Ii]nstruction|$)',
        r'(?:^|\n)(?:[Pp]air \d+[:.]?\s*)?(?:\d+\.?\s*)?[Qq]uestion:?\s*(.*?)\n+\s*[Aa]nswer:?\s*(.*?)(?=\n(?:[Pp]air \d+[:.]?\s*)?(?:\d+\.?\s*)?[Qq]uestion|$)',
        r'(?:^|\n)(?:\*\*)?(\d+)\.?\s*[Ii]nstruction:?(?:\*\*)?\s*(.*?)\n+\s*(?:\*\*)?[Rr]esponse(?: \1)?:?(?:\*\*)?\s*(.*?)(?=\n(?:\*\*)?(?:\d+)\.?\s*[Ii]nstruction|$)',
    ]
    
    pairs = []
    for pattern in patterns:
        matches = re.findall(pattern, file_content, re.DOTALL | re.IGNORECASE)
        for match in matches:
            if len(match) == 2:
                instruction, response = match
            elif len(match) == 3:
                _, instruction, response = match
            else:
                continue
            
            instruction = re.sub(r'\*\*|\*', '', instruction).strip()
            response = re.sub(r'\*\*|\*', '', response).strip()
            
            if instruction and response: 
                element={"instruction": instruction, "output": response}
                if element not in pairs:
                    pairs.append({"instruction": instruction, "output": response})
    return pairs


def read_xml_file(xml_file_path):
    with open(xml_file_path, 'r') as file:
        return file.read()


def save_json_file(json_data, output_file):
    with open(output_file, 'w') as file:
        json.dump(json_data, file, indent=4)


def create_sampled_datasets(json_data, output_folder, sample_sizes):
    for size in sample_sizes:
        if size > len(json_data):
            print(f"Warning: Requested sample size {size} is larger than the dataset. Using all available data.")
            sampled_data = json_data
        else:
            sampled_data = random.sample(json_data, size)
        
        output_file = os.path.join(output_folder, f'ehr_data_{size}.json')
        save_json_file(sampled_data, output_file)
        print(f"Created sample dataset with {len(sampled_data)} entries: {output_file}")

def trim_ehr(
    instruction: str,
    ehr:str,
    context_length: int,
    generation_length: int,
    model_name: str
):
    """
    Returns:
        a trimmed ehr string that fits within the context_length
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt_template = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n ### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    num_tokens_instruction = len(tokenizer.encode(instruction))
    num_tokens_prompt_template = len(tokenizer.encode(prompt_template))
        
    target_ehr_length = (
        context_length
        - generation_length
        - num_tokens_prompt_template
        - num_tokens_instruction
    )

    if target_ehr_length <= 0:
        return ""
    else:
        # Do a first pass with a fast tokenizer
        fast_tokenizer = tiktoken.get_encoding("cl100k_base")
        fast_encoded = fast_tokenizer.encode(ehr)
        fast_encoded_truncated = fast_encoded[-(2 * target_ehr_length) :]
        fast_truncated_ehr = fast_tokenizer.decode(fast_encoded_truncated)

        # Then do a second pass with the actual tokenizer
        encoded_ehr = tokenizer.encode(fast_truncated_ehr)
        truncated_encoded_ehr = encoded_ehr[-target_ehr_length:]
        truncated_ehr = tokenizer.decode(truncated_encoded_ehr)
        return truncated_ehr


def get_ehr_data(ehr_data_path, instruction_file, instruction_text, context_length):
    target_uid= instruction_file.split('.')[0].split('_')[0]
    context_window = instruction_file.split('.')[0].split('_')[-1]
    if not ehr_data_path.endswith('.jsonl'):
        xml_file_name = target_uid + '.xml'
        xml_file_path = os.path.join(ehr_data_path, xml_file_name)
        xml_content = read_xml_file(xml_file_path)
    else:
        xml_content = ""
        with open(ehr_data_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                if int(data.get('uid')) == int(target_uid) and int(data.get('window_index')) == int(context_window):
                    xml_content = data['text']
                    break
            else:
                print(f"Error: EHR data not found for instruction file: {instruction_file}")
    return xml_content
    


def process_files(instruction_folder, ehr_data_path, output_folder, context_length, json_folder=None):
    json_data = []
    problematic_files = []
    if json_folder!=None:
        print(f"Reading data from existing JSON file: {json_folder}")
        with open(json_folder, 'r') as file:
            json_data = json.load(file)
        save_json_file(json_data, os.path.join(output_folder, 'ehr_data.json'))
        return json_data
    instruction_files = os.listdir(instruction_folder)
    for instruction_file in tqdm(instruction_files, desc="Processing files"):
        if instruction_file.endswith('.txt'):
            instruction_file_path = os.path.join(instruction_folder, instruction_file)
            
            print(f"Processing file: {instruction_file}")
            
            with open(instruction_file_path, 'r') as file:
                instruction_content = file.read()
            
            pairs = parse_instruction_response_pairs(instruction_content)
            print(f"Number of instruction-response pairs found: {len(pairs)}")
            
            if len(pairs) == 0:
                problematic_files.append((instruction_file, instruction_content))
                continue

            xml_content = get_ehr_data(ehr_data_path, instruction_file, pairs[0]['instruction'], context_length)
            for pair in pairs:
                pair["input"] = xml_content
                json_data.append(pair)
        elif instruction_file.endswith('.json'):
            try:
                json_file_path = os.path.join(instruction_folder, instruction_file)
                with open(json_file_path, 'r') as file:
                    json_data_original = json.load(file)['response_data']
                    json_data_curr = json_data_original.copy()
                for instruction_pair in json_data_curr:
                    instruction=instruction_pair['instruction']
                    response = instruction_pair.get('output', instruction_pair.get('response'))
                    pair = {'instruction': instruction, 'output': response}
                    xml_content = get_ehr_data(ehr_data_path, instruction_file, instruction, context_length=16000)
                    pair["input"] = xml_content
                    pair["patient_id"] = instruction_file
                    json_data.append(pair)
            except Exception as e:
                print(f"Error processing file: {e}")
                print('Json object', json_data_curr)
                problematic_files.append([instruction_file, json_data_original, "parsing error"])
    print(f"Total number of objects in JSON data: {len(json_data)}")

    if problematic_files:
        problematic_files_path = os.path.join(output_folder, "problematic_files.txt")
        with open(problematic_files_path, 'w') as problematic_file:
            i=0
            for item in problematic_files:
                problematic_file.write(f"\n\n{'='*50}\n")
                problematic_file.write(f"File {i}: {item[0]}\n")
                problematic_file.write(f"{'='*50}\n")
                try:
                    if type(item[1]) == dict:
                        problematic_file.write(json.dumps(item[1], indent=4))
                    else:
                        problematic_file.write(str(item[1]))
                except:
                    problematic_file.write("Issue saving data as example")
                    continue
                i+=1
        print(f"Problematic files content saved to: {problematic_files_path}")
    else:
        print("No problematic files found.")
    save_json_file(json_data, os.path.join(output_folder, 'ehr_data.json'))
    return json_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instruction_folder', type=str)
    parser.add_argument('--ehr_data_path', type=str)
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--context_length', type=int, default=16384)
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    process_files(args.instruction_folder, args.ehr_data_path, args.output_folder, args.context_length)


if __name__ == "__main__":
    main()
