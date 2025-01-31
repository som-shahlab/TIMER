import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def calculate_bin_uniformity_score(bin_counts):
    """
    Calculate uniformity score for 10 bins.
    Returns coefficient of variation (lower is better).
    """
    # Handle pandas Series
    if isinstance(bin_counts, pd.Series):
        bin_counts_array = bin_counts.to_numpy()
    else:
        bin_counts_array = np.array(list(bin_counts.values()))
        
    if len(bin_counts_array) == 0 or np.mean(bin_counts_array) == 0:
        return float('inf')
        
    return np.std(bin_counts_array) / np.mean(bin_counts_array)

def match_instruction_ids(sampled_df, reference_file):
    """Match instructions with their IDs from the reference file."""
    ref_df = pd.read_csv(reference_file)
    instruction_to_id = dict(zip(ref_df['instruction'], ref_df['instruction_id']))
    sampled_df['instruction_id'] = sampled_df['instruction'].map(instruction_to_id)
    
    unmatched = sampled_df[sampled_df['instruction_id'].isna()]
    if len(unmatched) > 0:
        print(f"\nWarning: {len(unmatched)} instructions could not be matched with IDs.")
        print("First few unmatched instructions:")
        for inst in unmatched['instruction'].head().values:
            print(f"- {inst[:100]}...")
    
    return sampled_df

def create_10bin_uniform_sample(input_csv, reference_file, target_size=None, num_iterations=50):
    """
    Create a sample optimized for uniformity across 10 bins.
    """
    NUM_BINS = 10
    df = pd.read_csv(input_csv)
    
    # Create bins
    bin_edges = np.linspace(0, 1, NUM_BINS + 1)
    df['bin'] = pd.cut(df['relative_position'], 
                      bins=bin_edges, 
                      labels=range(NUM_BINS),  # Use numeric labels
                      include_lowest=True)
    
    # Calculate initial bin counts and target size
    initial_bin_counts = df['bin'].value_counts().sort_index()
    min_bin_count = initial_bin_counts.min()
    
    if target_size is None:
        samples_per_bin = min_bin_count
    else:
        samples_per_bin = target_size // NUM_BINS
    
    best_sampled_df = None
    best_uniformity_score = float('inf')
    
    # Create person to bin mapping
    person_bins = defaultdict(set)
    for _, row in df.iterrows():
        person_bins[row['person_id']].add(row['bin'])
    
    for iteration in range(num_iterations):
        sampled_indices = []
        bin_samples = defaultdict(int)
        used_indices = set()
        
        # First pass: Try to get diverse person coverage across bins
        persons = list(person_bins.keys())
        np.random.shuffle(persons)
        
        for person_id in persons:
            # Try to sample from each bin this person has data in
            for bin_idx in sorted(person_bins[person_id]):
                if bin_samples[bin_idx] >= samples_per_bin:
                    continue
                
                person_data = df[
                    (df['person_id'] == person_id) & 
                    (df['bin'] == bin_idx) & 
                    (~df.index.isin(used_indices))
                ]
                
                if len(person_data) > 0:
                    sampled_idx = np.random.choice(person_data.index)
                    sampled_indices.append(sampled_idx)
                    used_indices.add(sampled_idx)
                    bin_samples[bin_idx] += 1
        
        # Second pass: Fill remaining slots in each bin
        for bin_idx in range(NUM_BINS):
            remaining = samples_per_bin - bin_samples[bin_idx]
            if remaining <= 0:
                continue
            
            available = df[
                (df['bin'] == bin_idx) & 
                (~df.index.isin(used_indices))
            ]
            
            if len(available) > 0:
                additional_samples = np.random.choice(
                    available.index,
                    size=min(remaining, len(available)),
                    replace=False
                )
                sampled_indices.extend(additional_samples)
                bin_samples[bin_idx] += len(additional_samples)
        
        # Evaluate this attempt
        if sampled_indices:  # Check if we have any samples
            current_sampled_df = df.loc[sampled_indices].copy()
            # Create a complete bin count series with all bins
            bin_counts = pd.Series(0, index=range(NUM_BINS))
            current_counts = current_sampled_df['bin'].value_counts()
            bin_counts.update(current_counts)
            
            current_uniformity = calculate_bin_uniformity_score(bin_counts)
            
            if current_uniformity < best_uniformity_score:
                best_uniformity_score = current_uniformity
                best_sampled_df = current_sampled_df
                print(f"Iteration {iteration+1}: New best uniformity score: {current_uniformity:.4f}")
    
    if best_sampled_df is None:
        raise ValueError("Could not create a valid sample")
    
    # Match instruction IDs
    best_sampled_df = match_instruction_ids(best_sampled_df, reference_file)
    
    # Print statistics
    print("\nFinal Statistics:")
    print(f"Original samples: {len(df)}")
    print(f"Sampled: {len(best_sampled_df)}")
    print(f"Uniformity score (CV, lower is better): {best_uniformity_score:.4f}")
    print(f"Unique persons: {best_sampled_df['person_id'].nunique()}")
    print(f"Unique instructions: {best_sampled_df['instruction'].nunique()}")
    
    print("\nBin distribution:")
    final_bins = best_sampled_df['bin'].value_counts().sort_index()
    total_samples = len(best_sampled_df)
    for bin_idx in range(NUM_BINS):
        count = final_bins.get(bin_idx, 0)
        print(f"Bin {bin_idx+1}: {count} samples ({count/total_samples*100:.1f}%)")
    
    # Visualize the distributions
    plt.figure(figsize=(15, 5))
    
    # Original distribution
    plt.subplot(1, 3, 1)
    plt.hist(df['relative_position'], bins=NUM_BINS, density=True, alpha=0.6)
    plt.title('Original Distribution')
    plt.xlabel('Relative Position')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    
    # Sampled distribution
    plt.subplot(1, 3, 2)
    plt.hist(best_sampled_df['relative_position'], bins=NUM_BINS, density=True, alpha=0.6)
    plt.title(f'Sampled Distribution\n(n={len(best_sampled_df)})')
    plt.xlabel('Relative Position')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    
    # Bin counts
    plt.subplot(1, 3, 3)
    complete_counts = pd.Series(0, index=range(NUM_BINS))
    complete_counts.update(final_bins)
    plt.bar(range(1, NUM_BINS + 1), complete_counts, color='skyblue')
    plt.axhline(y=samples_per_bin, color='r', linestyle='--', label='Target per bin')
    plt.title('Samples per Bin')
    plt.xlabel('Bin')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('distribution_comparison.png')
    plt.close()
    
    # Drop the bin column and ensure instruction_id is first
    best_sampled_df = best_sampled_df.drop('bin', axis=1)
    cols = ['instruction_id'] + [col for col in best_sampled_df.columns if col != 'instruction_id']
    best_sampled_df = best_sampled_df[cols]
    
    return best_sampled_df

def main():
    input_csv = "../plot/temporal_distribution_bench.csv"
    reference_file = "../result/reviewed_eval_inference_input.csv"
    
    # Create optimized sample with 10 bins
    sampled_df = create_10bin_uniform_sample(
        input_csv,
        reference_file,
        target_size=None  # Will determine based on minimum bin count
    )
    
    # Save the sampled data
    output_path = "../plot/uniform_sampled_distribution.csv"
    sampled_df.to_csv(output_path, index=False)
    print(f"\nSaved optimized sample to '{output_path}'")
    print(f"Number of samples with instruction IDs: {sampled_df['instruction_id'].notna().sum()}")
    print(f"Number of samples missing instruction IDs: {sampled_df['instruction_id'].isna().sum()}")

if __name__ == "__main__":
    main()
