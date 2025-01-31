import json
import os
from datetime import datetime
import numpy as np
import argparse
from collections import defaultdict

def check_output_exists(output_dir):
    """Check if output directory exists and contains JSON files"""
    if not output_dir:
        return False
    if not os.path.exists(output_dir):
        return False
    # Check if directory contains any JSON files
    return any(f.endswith('.json') for f in os.listdir(output_dir))

def get_person_id_from_filename(filename):
    """Extract person_id from filename consistently"""
    if filename.endswith('.json'):
        return str(filename.split('_')[0])
    return None

def get_existing_person_ids(output_dirs):
    """Get person_ids from existing output directories with detailed logging"""
    person_ids = set()
    for output_dir in output_dirs:
        if output_dir and os.path.exists(output_dir):
            print(f"\nChecking directory: {output_dir}")
            json_files = [f for f in os.listdir(output_dir) if f.endswith('.json') and not f == 'sampling_log.txt']
            print(f"Found {len(json_files)} JSON files")
            
            for filename in json_files:
                person_id = get_person_id_from_filename(filename)
                if person_id:
                    person_ids.add(person_id)
    
    print(f"\nTotal unique person_ids found: {len(person_ids)}")
    if len(person_ids) > 0:
        print(f"Sample person_ids: {list(person_ids)[:5]}")
    return person_ids

def collect_and_process_pairs(input_folder, last_25_output=None, last_10_output=None,
                            span_100_output=None, balanced_100_output=None, 
                            num_bins=10, sample_size=5000):
    """Process pairs with dataset-level sampling and temporal distribution control"""
    # Check which outputs need processing
    outputs_to_process = []
    if last_25_output and not check_output_exists(last_25_output):
        outputs_to_process.append(('last_25', last_25_output))
    if span_100_output and not check_output_exists(span_100_output):
        outputs_to_process.append(('span_100', span_100_output))
    if balanced_100_output and not check_output_exists(balanced_100_output):
        outputs_to_process.append(('balanced_100', balanced_100_output))
    if last_10_output and not check_output_exists(last_10_output):
        outputs_to_process.append(('last_10', last_10_output))
    
    if not outputs_to_process:
        print("All output folders exist and contain data. Nothing to process.")
        return

    print("\nCollecting valid pairs...")
    all_valid_pairs = []  # Store all valid pairs across files
    file_to_pairs = defaultdict(list)  # Track which pairs belong to which file
    
    # Track unique person_ids in input data
    input_person_ids = set()
    
    # First, collect all input person_ids from filenames
    for filename in os.listdir(input_folder):
        person_id = get_person_id_from_filename(filename)
        if person_id:
            input_person_ids.add(person_id)
    
    # Collect all valid pairs while maintaining file association
    for filename in os.listdir(input_folder):
        if not filename.endswith('.json'):
            continue
        
        person_id = get_person_id_from_filename(filename)
        if not person_id:
            continue
            
        with open(os.path.join(input_folder, filename), 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue
            
        if 'error' in data:
            continue
        
        try:
            start_date = datetime.strptime(data['start_date'], '%Y-%m-%d')
            end_date = datetime.strptime(data['end_date'], '%Y-%m-%d')
            window_duration = (end_date - start_date).days
            
            if window_duration == 0:
                continue
                
            response_data = data['response_data']
            if isinstance(response_data, dict):
                response_data = [response_data]
            
            file_pairs = []
            for item in response_data:
                if isinstance(item, dict) and 'error' in item:
                    continue
                    
                try:
                    evidence_date = datetime.strptime(item['evidence'], '%m/%d/%Y')
                    
                    if evidence_date < start_date or evidence_date > end_date:
                        continue
                        
                    relative_position = (evidence_date - start_date).days / window_duration
                    relative_position = max(0, min(1, relative_position))
                    
                    pair_data = {
                        'person_id': person_id,
                        'window_index': data['window_index'],
                        'start_date': data['start_date'],
                        'end_date': data['end_date'],
                        'instruction': item['instruction'],
                        'response': item['response'],
                        'evidence': item['evidence'],
                        'relative_position': relative_position,
                        'source_file': filename
                    }
                    
                    file_pairs.append(pair_data)
                    all_valid_pairs.append(pair_data)
                        
                except (KeyError, ValueError):
                    continue
            
            if file_pairs:
                file_to_pairs[filename].extend(file_pairs)
                    
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            continue
    
    print(f"\nTotal JSON files with valid pairs: {len(file_to_pairs)}")
    print(f"Total valid pairs across all files: {len(all_valid_pairs)}")
    print(f"Total unique person_ids in input data: {len(input_person_ids)}")
    
    if len(file_to_pairs) < 1:
        raise ValueError("No valid files found")

    # Get person_ids from existing outputs if we need to sample last 10%
    existing_person_ids = None
    if ('last_10', last_10_output) in outputs_to_process:
        print("\nCollecting existing person_ids for last 10% sampling...")
        existing_output_dirs = [d for d in [last_25_output, span_100_output, balanced_100_output] 
                              if d and os.path.exists(d)]
        existing_person_ids = get_existing_person_ids(existing_output_dirs)
        
        # Debug info about person_ids
        print("\nPerson ID Analysis:")
        print(f"Input person_ids: {len(input_person_ids)}")
        print(f"Existing person_ids: {len(existing_person_ids)}")
        overlap = input_person_ids.intersection(existing_person_ids)
        print(f"Overlapping person_ids: {len(overlap)}")
        if len(overlap) > 0:
            print("Sample overlapping person_ids:", list(overlap)[:5])
        else:
            print("\nWARNING: No overlap between input and existing person_ids!")
            print("Sample input person_ids:", list(input_person_ids)[:5])
            print("Sample existing person_ids:", list(existing_person_ids)[:5])
    
    # Perform sampling for each method that needs processing
    for output_type, output_dir in outputs_to_process:
        if output_type == 'last_25':
            process_last_25_sampling(all_valid_pairs, file_to_pairs, sample_size, output_dir)
        elif output_type == 'last_10':
            process_last_n_sampling(all_valid_pairs, file_to_pairs, sample_size, output_dir, 
                                  threshold=0.9, name="10", existing_person_ids=existing_person_ids)
        elif output_type == 'span_100':
            process_span_sampling(all_valid_pairs, file_to_pairs, sample_size, output_dir)
        elif output_type == 'balanced_100':
            process_balanced_sampling(all_valid_pairs, file_to_pairs, sample_size, output_dir, num_bins)


def process_last_25_sampling(all_pairs, file_to_pairs, sample_size, output_dir):
    """Sample from the last quarter at dataset level while ensuring sample size"""
    print("\nProcessing last 25% selection...")
    
    # Get pairs from last quarter across all files
    last_quarter_pairs = [pair for pair in all_pairs if pair['relative_position'] >= 0.75]
    print(f"Pairs in last quarter: {len(last_quarter_pairs)}")
    
    if len(last_quarter_pairs) < sample_size:
        raise ValueError(f"Not enough pairs in last quarter. Found {len(last_quarter_pairs)}, need {sample_size}")
    
    # Count how many files have last quarter data
    files_with_last_quarter = set()
    for filename, file_pairs in file_to_pairs.items():
        if any(p['relative_position'] >= 0.75 for p in file_pairs):
            files_with_last_quarter.add(filename)
    
    print(f"Files with last quarter data: {len(files_with_last_quarter)} out of {len(file_to_pairs)}")
    
    # Sample pairs from files with last quarter data
    sampled_pairs = []
    files_covered = set()
    
    # First, sample one pair from each file that has last quarter pairs
    for filename in files_with_last_quarter:
        file_pairs = [p for p in file_to_pairs[filename] if p['relative_position'] >= 0.75]
        if file_pairs:
            selected_pair = np.random.choice(file_pairs)
            sampled_pairs.append(selected_pair)
            files_covered.add(filename)
    
    # Then sample remaining pairs randomly from last quarter to reach sample_size
    remaining_needed = sample_size - len(sampled_pairs)
    if remaining_needed > 0:
        # Include ALL last quarter pairs except those already sampled
        remaining_pairs = [p for p in last_quarter_pairs if p not in sampled_pairs]
        if remaining_pairs:
            additional_samples = np.random.choice(
                remaining_pairs,
                min(remaining_needed, len(remaining_pairs)),
                replace=False
            )
            sampled_pairs.extend(additional_samples)
    
    save_sampled_pairs(sampled_pairs, output_dir)
    save_sampling_log(output_dir, "last_25", len(last_quarter_pairs), len(sampled_pairs))


def process_span_sampling(all_pairs, file_to_pairs, sample_size, output_dir):
    """Sample randomly across whole span while ensuring file coverage"""
    print("\nProcessing random span selection...")
    
    sampled_pairs = []
    files_covered = set()
    
    # First, sample one pair from each file
    for filename, file_pairs in file_to_pairs.items():
        selected_pair = np.random.choice(file_pairs)
        sampled_pairs.append(selected_pair)
        files_covered.add(filename)
    
    # Then sample remaining pairs randomly to reach sample_size
    remaining_needed = sample_size - len(sampled_pairs)
    if remaining_needed > 0:
        # Here's the key change: Include ALL pairs in the sampling pool
        remaining_pairs = [p for p in all_pairs if p not in sampled_pairs]
        if remaining_pairs:
            additional_samples = np.random.choice(
                remaining_pairs,
                min(remaining_needed, len(remaining_pairs)),
                replace=False
            )
            sampled_pairs.extend(additional_samples)
    
    save_sampled_pairs(sampled_pairs, output_dir)
    save_sampling_log(output_dir, "random_span", len(all_pairs), len(sampled_pairs))


def process_balanced_sampling(all_pairs, file_to_pairs, sample_size, output_dir, num_bins):
    """Sample equally from temporal bins while ensuring file coverage"""
    print("\nProcessing balanced selection...")
    
    # Calculate target samples per bin
    target_per_bin = sample_size // num_bins
    print(f"Target samples per bin: {target_per_bin}")
    
    # Distribute all pairs into bins
    bin_edges = np.linspace(0, 1, num_bins + 1)
    binned_pairs = defaultdict(list)
    
    for pair in all_pairs:
        bin_idx = np.digitize(pair['relative_position'], bin_edges) - 1
        if bin_idx == num_bins:
            bin_idx = num_bins - 1
        binned_pairs[bin_idx].append(pair)
    
    # First pass: Try to get one pair from each file, distributed across bins
    sampled_pairs = []
    files_covered = set()
    current_bin_counts = defaultdict(int)
    
    # Shuffle files to randomize selection
    files = list(file_to_pairs.keys())
    np.random.shuffle(files)
    
    for filename in files:
        file_pairs = file_to_pairs[filename]
        # Group file pairs by bin
        file_bin_pairs = defaultdict(list)
        for pair in file_pairs:
            bin_idx = np.digitize(pair['relative_position'], bin_edges) - 1
            if bin_idx == num_bins:
                bin_idx = num_bins - 1
            file_bin_pairs[bin_idx].append(pair)
        
        # Find bin with lowest count that has pairs from this file
        valid_bins = [
            bin_idx for bin_idx, pairs in file_bin_pairs.items()
            if current_bin_counts[bin_idx] < target_per_bin
        ]
        
        if valid_bins:
            selected_bin = min(valid_bins, key=lambda x: current_bin_counts[x])
            selected_pair = np.random.choice(file_bin_pairs[selected_bin])
            sampled_pairs.append(selected_pair)
            files_covered.add(filename)
            current_bin_counts[selected_bin] += 1
    
    # Second pass: Fill up each bin to reach target_per_bin
    for bin_idx in range(num_bins):
        remaining_needed = target_per_bin - current_bin_counts[bin_idx]
        if remaining_needed <= 0:
            continue
            
        # Include ALL pairs in this bin except those already sampled
        eligible_pairs = [
            p for p in binned_pairs[bin_idx]
            if p not in sampled_pairs  # Only exclude already sampled pairs
        ]
        
        if eligible_pairs:
            additional_samples = np.random.choice(
                eligible_pairs,
                min(remaining_needed, len(eligible_pairs)),
                replace=False
            )
            sampled_pairs.extend(additional_samples)
            current_bin_counts[bin_idx] += len(additional_samples)
    
    # Print bin distribution
    print("\nFinal bin distribution:")
    for bin_idx in range(num_bins):
        bin_start = bin_edges[bin_idx]
        bin_end = bin_edges[bin_idx + 1]
        print(f"Bin {bin_start:.2f}-{bin_end:.2f}: {current_bin_counts[bin_idx]} pairs")
    
    save_sampled_pairs(sampled_pairs, output_dir)
    save_sampling_log(output_dir, "balanced_span", len(all_pairs), len(sampled_pairs),
                     num_bins=num_bins, bin_distribution=current_bin_counts)


def process_last_n_sampling(all_pairs, file_to_pairs, sample_size, output_dir, threshold=0.9, name="10", existing_person_ids=None):
    """Sample from the last N% at dataset level while ensuring sample size"""
    print(f"\nProcessing last {name}% selection...")
    
    # Get pairs from last N% across all files
    last_n_pairs = [pair for pair in all_pairs if pair['relative_position'] >= threshold]
    print(f"Pairs in last {name}%: {len(last_n_pairs)}")
    
    # Print sample of pairs before filtering
    if len(last_n_pairs) > 0:
        print("\nSample pairs before filtering:")
        for pair in last_n_pairs[:3]:
            print(f"Person ID: {pair['person_id']}, Relative Position: {pair['relative_position']}")
    
    # Filter pairs based on existing_person_ids if provided
    if existing_person_ids:
        last_n_pairs = [pair for pair in last_n_pairs if pair['person_id'] in existing_person_ids]
        print(f"Pairs after filtering for existing person_ids: {len(last_n_pairs)}")
        
        if len(last_n_pairs) > 0:
            print("\nSample pairs after filtering:")
            for pair in last_n_pairs[:3]:
                print(f"Person ID: {pair['person_id']}, Relative Position: {pair['relative_position']}")
    
    if len(last_n_pairs) < sample_size:
        raise ValueError(f"Not enough pairs in last {name}%. Found {len(last_n_pairs)}, need {sample_size}")
    
    # Count how many files have last N% data
    files_with_last_n = {pair['source_file'] for pair in last_n_pairs}
    print(f"Files with last {(1-threshold)*100:.0f}% data: {len(files_with_last_n)} out of {len(file_to_pairs)}")
    
    # Sample pairs from files with last N% data
    sampled_pairs = []
    files_covered = set()
    
    # First, sample one pair from each file that has last N% pairs
    for filename in files_with_last_n:
        file_pairs = [p for p in file_to_pairs[filename] if p['relative_position'] >= threshold]
        if file_pairs and (not existing_person_ids or any(p['person_id'] in existing_person_ids for p in file_pairs)):
            eligible_pairs = [p for p in file_pairs if not existing_person_ids or p['person_id'] in existing_person_ids]
            if eligible_pairs:
                selected_pair = np.random.choice(eligible_pairs)
                sampled_pairs.append(selected_pair)
                files_covered.add(filename)
    
    # Then sample remaining pairs randomly from last N% to reach sample_size
    remaining_needed = sample_size - len(sampled_pairs)
    if remaining_needed > 0:
        remaining_pairs = [p for p in last_n_pairs if p not in sampled_pairs]
        if remaining_pairs:
            additional_samples = np.random.choice(
                remaining_pairs,
                min(remaining_needed, len(remaining_pairs)),
                replace=False
            )
            sampled_pairs.extend(additional_samples)
    
    save_sampled_pairs(sampled_pairs, output_dir)
    save_sampling_log(output_dir, f"last_{(1-threshold)*100:.0f}", len(last_n_pairs), len(sampled_pairs))


def save_sampled_pairs(pairs, output_dir):
    """Save the sampled pairs grouped by person_id and window_index"""
    # Group by person and window
    grouped_pairs = defaultdict(lambda: defaultdict(list))
    for pair in pairs:
        grouped_pairs[pair['person_id']][pair['window_index']].append({
            'instruction': pair['instruction'],
            'response': pair['response'],
            'evidence': pair['evidence']
        })
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each person's windows
    for person_id, windows in grouped_pairs.items():
        for window_index, response_data in windows.items():
            # Get any pair for this window to get dates
            sample_pair = next(p for p in pairs 
                             if p['person_id'] == person_id and p['window_index'] == window_index)
            
            filename = f"{person_id}_{window_index}.json"
            data = {
                'person_id': person_id,
                'window_index': window_index,
                'start_date': sample_pair['start_date'],
                'end_date': sample_pair['end_date'],
                'response_data': response_data
            }
            
            with open(os.path.join(output_dir, filename), 'w') as f:
                json.dump(data, f, indent=2)
    
    unique_persons = len(grouped_pairs)
    unique_windows = sum(len(windows) for windows in grouped_pairs.values())
    print(f"Saved data for {unique_persons} persons ({unique_windows} windows) to {output_dir}")

def save_sampling_log(output_dir, selection_type, total_pairs, sampled_pairs, 
                     num_bins=None, bin_distribution=None):
    """Save sampling log with relevant information"""
    log_content = f"""Sampling Information:
Selection type: {selection_type}
Total available pairs: {total_pairs}
Sampled pairs: {sampled_pairs}
"""
    
    if bin_distribution is not None and isinstance(bin_distribution, dict):
        log_content += f"\nBin distribution:"
        bin_edges = np.linspace(0, 1, num_bins + 1)
        for bin_idx in range(num_bins):
            bin_start = bin_edges[bin_idx]
            bin_end = bin_edges[bin_idx + 1]
            bin_count = bin_distribution[bin_idx]
            log_content += f"\nBin {bin_start:.1f}-{bin_end:.1f}: {bin_count} pairs"
    
    with open(os.path.join(output_dir, 'sampling_log.txt'), 'w') as f:
        f.write(log_content)

def parse_args():
    parser = argparse.ArgumentParser(description='Sample pairs from JSON files based on temporal distribution')
    parser.add_argument('--input_folder', required=True, help='Input folder containing JSON files')
    parser.add_argument('--last_25_output', help='Output folder for last 25% samples')
    parser.add_argument('--last_10_output', help='Output folder for last 10% samples')
    parser.add_argument('--span_100_output', help='Output folder for random whole window samples')
    parser.add_argument('--balanced_100_output', help='Output folder for balanced whole window samples')
    parser.add_argument('--num_bins', type=int, default=10, help='Number of bins for balanced sampling')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try:
        collect_and_process_pairs(
            args.input_folder,
            args.last_25_output,
            args.last_10_output,
            args.span_100_output,
            args.balanced_100_output,
            args.num_bins
        )
        print("\nSuccessfully completed sampling")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
