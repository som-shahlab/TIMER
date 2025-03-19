import numpy as np
import argparse
import json
import glob
import os

def extract_scores(data, criterion):
    return [
        entry["evaluation"][criterion]["score"]
        for entry in data["detailed_results"]
    ]

def return_bootstrap(judge_outputs:str, metric:str):
    # Example: Extract correctness scores
    with open(judge_outputs, 'r') as file:
            data = json.load(file)
    scores = extract_scores(data, metric)

    # Perform bootstrap analysis
    n_bootstraps = 10000
    bootstrap_means = []

    for _ in range(n_bootstraps):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        bootstrap_means.append(np.mean(sample))

    # Compute confidence intervals
    lower_bound = np.percentile(bootstrap_means, 2.5)
    upper_bound = np.percentile(bootstrap_means, 97.5)

    # Output results
    print(f"Mean correctness score: {np.mean(scores):.3f}")
    gap=(upper_bound-lower_bound)/2
    print(f"Confidence interval gap: {gap:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--judge_filepath', required=False, type=str)
    parser.add_argument('--metric', required=False, type=str, default="correctness") #or correctness
    args = parser.parse_args()
    user="name"

    if args.judge_filepath:
        return_bootstrap(args.judge_filepath, args.metric)
        directory = "../result/judge_outputs"
        for filepath in glob.glob(os.path.join(directory, '*')):
            file_name=filepath.split('/')[-1]
            print(f"Processing file: {file_name}")
            return_bootstrap(filepath, args.metric)
