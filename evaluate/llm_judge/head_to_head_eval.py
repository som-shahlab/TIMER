import argparse
import json
import os
import pandas as pd
import requests
import time
from typing import Dict, Any

# Azure OpenAI settings
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DEPLOYMENT_NAME = "gpt-4o-mini"
API_VERSION = "2023-05-15"

class LLMEvalPromptGenerator:
    instruct_prompt = '''Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. 
Your evaluation should consider correctness and helpfulness. You will be given a reference answer, assistant A's answer, and assistant B's answer. Your job is to evaluate which assistant's answer is better. Begin your evaluation by comparing both assistants' answers with the reference answer. Identify and correct any mistakes. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. 
Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. 
After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie. You must begin with [[A]] or [[B]] or [[C]].
 Assigning "[[C]]" should be a very last resort! used only if you absolutely cannot discern any difference in the quality of the two responses.'''

    role = 'Assistant'

    @staticmethod
    def conv_to_str(question, ground_truth, ans1, ans2):
        return (f'[Context]\n'
                f'[Question]\n{question}\n\n'
                f'[Ground Truth]\n{ground_truth}\n\n'
                f'[{LLMEvalPromptGenerator.role} A]\n{ans1}\n\n[End of {LLMEvalPromptGenerator.role} 1]\n\n'
                f'[{LLMEvalPromptGenerator.role} B]\n{ans2}\n\n[End of {LLMEvalPromptGenerator.role} 2]\n\n')

    @staticmethod
    def compare_messages_gen(sample):
        messages = [
            {"role": "system", "content": LLMEvalPromptGenerator.instruct_prompt}
        ]
        messages.append({
            "role": "user", 
            "content": LLMEvalPromptGenerator.conv_to_str(
                sample['question'], 
                sample['ground_truth'], 
                sample['ans1'], 
                sample['ans2']
            )
        })
        return messages

def completion_with_backoff(**kwargs) -> Dict[str, Any]:
    retry_count = 0
    while True:
        retry_count += 1
        try:
            url = f"{OPENAI_API_BASE}/deployments/{DEPLOYMENT_NAME}/chat/completions?api-version={API_VERSION}"
            headers = {
                "Ocp-Apim-Subscription-Key": OPENAI_API_KEY,
                "Content-Type": 'application/json'
            }
            data = {
                "messages": kwargs['messages'],
                "max_tokens": kwargs.get('max_tokens', 1000),
                "temperature": kwargs.get('temperature', 0.7)
            }
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as error:
            print(f"Error: {error}")
            if retry_count > 30:
                return {}
            time.sleep(10)

def call_async(batch, message_generator):
    results = []
    for sample in batch:
        messages = message_generator(sample)
        response = completion_with_backoff(messages=messages)
        if response and 'choices' in response:
            result = {
                'question_id': sample.get('question_id'),
                'result': response['choices'][0]['message']['content'],
                'order': sample.get('order', 'original')
            }
            results.append(result)
    return results

class ChatEvaluation:
    @staticmethod
    def eval(samples):
        win1 = 0
        win2 = 0
        tie = 0

        # Loop through each sample to analyze the verdict
        for sample in samples:
            result = sample['result']
            order = sample['order']
            verdict = None

            # Search for the final verdict in the result string
            if '[[A]]' in result:
                verdict = 'A'
                if order == 'original':
                    win1 += 1
                else:
                    win2 += 1
            elif '[[B]]' in result:
                verdict = 'B'
                if order == 'original':
                    win2 += 1
                else:
                    win1 += 1
            elif '[[C]]' in result:
                verdict = 'C'
                tie += 1
            else:
                print(f"Warning: Verdict not found in result for question_id {sample.get('question_id')}. Result: {result}")

        # Calculate rates
        total = win1 + win2 + tie
        if total > 0:
            win1_rate = win1 / total
            win2_rate = win2 / total
            tie_rate = tie / total
        else:
            win1_rate = win2_rate = tie_rate = 0.0

        print(f"Processed {len(samples)} Samples, Rates: Win1: {win1_rate:.3f}, Win2: {win2_rate:.3f}, Tie: {tie_rate:.3f}")

        return win1, win2, tie

def process_batch(batch, results, processed_ids, args):
    try:
        batch_results = call_async(batch, LLMEvalPromptGenerator.compare_messages_gen)
        if batch_results:
            results.extend(batch_results)
            processed_ids.update(r['question_id'] for r in batch_results)
            
            win1, win2, tie = ChatEvaluation.eval(results)
            aggregated_metrics = {
                "win1_rate": win1 / (win1 + win2 + tie),
                "win2_rate": win2 / (win1 + win2 + tie),
                "tie_rate": tie / (win1 + win2 + tie),
                "processed_samples": len(results)
            }
            
            with open(args.output_file, 'w') as f:
                for result in results:
                    f.write(json.dumps(result) + '\n')
                f.write(json.dumps({"aggregated_metrics": aggregated_metrics}, indent=4))
    except Exception as e:
        print(f"Error processing batch: {e}")
    return results, processed_ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_a_responses', required=True, type=str)
    parser.add_argument('--model_b_responses', required=True, type=str)
    parser.add_argument('--reference_answers', required=True, type=str)
    parser.add_argument('--output_file', required=True, type=str)

    args = parser.parse_args()

    reference_data = pd.read_csv(args.reference_answers)
    model_a_data = pd.read_csv(args.model_a_responses)
    model_b_data = pd.read_csv(args.model_b_responses)
    
    reference_data_annotator1 = reference_data[reference_data['annotator_num'] == 'Annotator_1']
    
    # Create samples with original order
    samples_original = []
    # Create samples with swapped order
    samples_swapped = []
    
    for _, ref_row in reference_data_annotator1.iterrows():
        instruction_id = ref_row['instruction_id']
        if instruction_id in model_a_data['instruction_id'].values and instruction_id in model_b_data['instruction_id'].values:
            model_a_response = model_a_data[model_a_data['instruction_id'] == instruction_id]['model_response'].iloc[0]
            model_b_response = model_b_data[model_b_data['instruction_id'] == instruction_id]['model_response'].iloc[0]
            
            # Original order sample
            samples_original.append({
                'question_id': f"{instruction_id}_original",
                'question': ref_row['question'],
                'ground_truth': ref_row['clinician_response'],
                'ans1': model_a_response,
                'ans2': model_b_response,
                'order': 'original'
            })
            
            # Swapped order sample
            samples_swapped.append({
                'question_id': f"{instruction_id}_swapped",
                'question': ref_row['question'],
                'ground_truth': ref_row['clinician_response'],
                'ans1': model_b_response,
                'ans2': model_a_response,
                'order': 'swapped'
            })

    # Combine all samples
    all_samples = samples_original + samples_swapped
    print(f"Total samples prepared: {len(all_samples)} (including swapped order)")

    results = []
    BATCH_SIZE = 1
    processed_ids = set()
    
    for i in range(30):
        batch = []
        for sample in all_samples:
            if sample['question_id'] in processed_ids:
                continue
            batch.append(sample)
            if len(batch) >= BATCH_SIZE:
                results, processed_ids = process_batch(batch, results, processed_ids, args)
                batch = []

        if batch:
            results, processed_ids = process_batch(batch, results, processed_ids, args)

    # Calculate final metrics
    win1, win2, tie = ChatEvaluation.eval(results)
    aggregated_metrics = {
        "win1_rate": win1 / (win1 + win2 + tie),
        "win2_rate": win2 / (win1 + win2 + tie),
        "tie_rate": tie / (win1 + win2 + tie),
        "processed_samples": len(results),
        "unique_questions": len(results) // 2  # Since each question is evaluated twice
    }
    
    with open(args.output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
        f.write(json.dumps({"aggregated_metrics": aggregated_metrics}, indent=4))

    print(f"\nFinal results saved to: {args.output_file}")

if __name__ == "__main__":
    main()
