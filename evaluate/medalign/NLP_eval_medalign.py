
import argparse
import logging
import os
from typing import List, Optional, Tuple, Union

import evaluate
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt_tab')

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate NLG metrics (eg BERTScore, BLEU) for a CSV file "
            "containing prompts "
        )
    )
    parser.add_argument(
        "--path_to_reference_responses",
        # required=True,
        type=str,
        default= "../evaluate/temporal/synth_eval_reference_answers.csv", 
        help="Path to the CSV file containing clinician-generated responses",
    )
    parser.add_argument(
        "--path_to_model_responses",
        # required=True,
        type=str,
        default="../medalign_inference_results/",
        help=(
            "Path to CSV file containing model-generated responses OR "
            "path to a directory containing such files"
        ),
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="eval_results.csv",
        help="Name of the output CSV file for storing metrics",
    )
    parser.add_argument(
        "--model_name_col",
        type=str,
        default="model_name",
        help="Name of column where the model name is stored",
    )
    parser.add_argument(
        "--model_response_col",
        type=str,
        default="model_response",
        help="Name of column where model responses are stored",
    )
    parser.add_argument(
        "--reference_response_col",
        type=str,
        default="clinician_response",
        help="Column where clinician-generated reference responses are stored",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["bertscore", "meteor", "chrf", "google_bleu", "rouge"],
        help="List of evaluation metrics to calculate",
    )
    args = parser.parse_args()  # Parse the command-line arguments
    # main(args)
    return args


class Evaluator:
    def __init__(
        self,
        metrics: list,
        verbose: bool = False,
    ):
        self.metrics = metrics
        self.verbose = verbose
        self.available_metrics = [
            "bertscore",
            "rouge",
            "bleu",
            "meteor",
            "bleurt",
            "google_bleu",
            "chrf",
            "comet",
        ]
        self.loaded_metrics = {}

    def process_metric(self, dict_: dict, metric_name: str) -> float:
        """Evaluators return a dict, we want to extract the float from this dictionary"""
        if metric_name == "bleu":
            return float(dict_["bleu"])
        elif metric_name == "rouge":
            return float(dict_["rougeL"])
        elif metric_name == "meteor":
            return float(dict_["meteor"])
        elif metric_name == "bertscore":
            assert len(dict_["f1"]) == 1
            return dict_["f1"][0]
        elif metric_name == "bleurt":
            return float(np.array(dict_["scores"]))
        elif metric_name == "google_bleu":
            return float(dict_["google_bleu"])
        elif metric_name == "chrf":
            return float(dict_["score"])
        elif metric_name == "comet":
            if isinstance(dict_["comet"], list):
                return dict_["comet"]
            return float(dict_["comet"])
        else:
            raise ValueError(f"Metric name '{metric_name}' not supported!")

    def load_metrics(self):
        """Load metrics so we can evalaute everything"""
        for metric_name in self.metrics:
            if metric_name == "bleurt":
                self.loaded_metrics[metric_name] = evaluate.load(
                    "bleurt", "bleurt-base-512"
                )
            else:
                self.loaded_metrics[metric_name] = evaluate.load(metric_name)

    def evaluate_scores(self, references: List[str], predictions: List[str]):
        selected_metrics = [m for m in self.metrics if m in self.available_metrics]
        sourceless_metrics = {}

        # TODO: Improve error handling for case where prediction is null
        predictions = [x if not pd.isnull(x) else "" for x in predictions]

        self.load_metrics()
        for metric_name, metric in self.loaded_metrics.items():
            sourceless_metrics[metric_name] = metric

        if not selected_metrics:
            print(
                f"No supported metrics found!\n"
                f"Metrics provided: {self.metrics}\n"
                f"Metrics Supported: {self.available_metrics}"
            )
            return {}

        is_ref_pred_eq = len(references) == len(predictions)
        if not is_ref_pred_eq:
            raise ValueError("references and predictions must be of equal length.")

        metrics = {}
        for metric_name, metric_obj in sourceless_metrics.items():
            if metric_name == "bertscore":
                metrics[metric_name] = metric_obj.compute(
                    predictions=predictions,
                    references=references,
                    model_type="distilbert-base-uncased",
                )["f1"]
            elif metric_name == "chrf":
                metrics[metric_name] = []
                for pred, ref in zip(predictions, references):
                    metric_value = metric_obj.compute(
                        predictions=[pred], references=[ref], word_order=2
                    )["score"]
                    metrics[metric_name].append(metric_value)
            elif metric_name == "google_bleu":
                metrics[metric_name] = []
                for pred, ref in zip(predictions, references):
                    metric_value = metric_obj.compute(
                        predictions=[pred], references=[ref]
                    )["google_bleu"]
                    metrics[metric_name].append(metric_value)
            elif metric_name == "rouge":
                metrics[metric_name] = []
                for pred, ref in zip(predictions, references):
                    metric_value = metric_obj.compute(
                        predictions=[pred],
                        references=[ref],
                        tokenizer=lambda s: s.split(),  # TODO: Note that we used the default tokenizer in the original manuscript
                    )["rougeL"]
                    metrics[metric_name].append(metric_value)
            elif metric_name == "meteor":
                metrics[metric_name] = []
                for pred, ref in zip(predictions, references):
                    metric_value = metric_obj.compute(
                        predictions=[pred], references=[ref]
                    )["meteor"]
                    metrics[metric_name].append(metric_value)
            else:
                raise ValueError(f"Processing not supported for metric = {metric_name}")

        return metrics


def process_file(file_path: str, args: argparse.Namespace) -> None:
    """
    Processes a single file for evaluation.

    Parameters:
    file_path (str): Path to the file to be processed.
    args (argparse.Namespace): Command line arguments passed to the script.

    Returns:
    None
    """
    # TODO: We should be able to just read this in from `merged_df`
    model_name = "Meta-Llama-3-8B-Instruct"
    prompt_template_name = "generic"

    gold_df = pd.read_csv(args.path_to_reference_responses)
    gold_df = gold_df.query("annotator_num == 'Annotator_1'")
    model_output_df = pd.read_csv(file_path)
    merged_df = gold_df.merge(model_output_df, on="instruction_id", how="inner")
    print('MERGED SIZE:', merged_df.shape)
    print('GOLD SIZE:', gold_df['instruction_id'])
    print('MODEL SIZE:', model_output_df['instruction_id'])

    evaluator = Evaluator(metrics=args.metrics)
    scores = evaluator.evaluate_scores(
        references=merged_df[args.reference_response_col].tolist(),
        predictions=merged_df[args.model_response_col].tolist(),
        # n_source_toks=args.n_source_tokens,
    )
    results_df = pd.DataFrame(scores)

    for col in [
        "instruction_id",
        # "instruction",
        # "context_length",
        "generation_length",
        "is_applicable",
        "is_sufficient",
    ]:
        results_df[col] = merged_df[col]

    # TODO: Can move these two lines into the above loop, should now be handled for you by `inference.py`
    results_df["model_name"] = model_name
    results_df["prompt_template_name"] = prompt_template_name

    grouped_results = results_df.groupby(
        [
            "model_name",
            "prompt_template_name",
            # "context_length",
            "generation_length",
        ]
    )[args.metrics].mean()
    
    results_str = grouped_results.to_string()
    filename = os.path.basename(args.model_name_col) + ".txt"
    
    directory = "../medalign-clean/src/medalign/results"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, filename)
    with open(file_path, "w") as file:
        file.write(results_str)

    return results_df

def boostrap_results(results_df: pd.DataFrame, metric: str) -> Tuple[float, float]:
    """
    Perform bootstrap analysis on the results.

    Parameters:
    results_df (pd.DataFrame): DataFrame containing the results to be bootstrapped.
    metric (str): Name of the metric to be bootstrapped.

    Returns:
    Tuple[float, float]: Tuple containing the lower and upper bounds of the confidence interval.
    """
    total_mean = results_df.groupby(
        [
            "model_name",
            "prompt_template_name",
            # "context_length",
            "generation_length",
        ]
    )[metric].mean()
    scores = results_df[metric].tolist()
    n_bootstraps = 10000
    bootstrap_means = []

    for _ in range(n_bootstraps):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        bootstrap_means.append(np.mean(sample))

    lower_bound = np.percentile(bootstrap_means, 2.5)
    upper_bound = np.percentile(bootstrap_means, 97.5)

    return lower_bound, upper_bound, total_mean


def main():
    args = parse_args()
    if os.path.isdir(args.path_to_model_responses):
        # If path given is directory, process each .csv file w/in the directory
        all_results_dfs = []
        for i, filename in enumerate(sorted(os.listdir(args.path_to_model_responses))):
            if filename.endswith(".csv"):
                file_path = os.path.join(args.path_to_model_responses, filename)
                print(f"Processing {file_path}...")
                all_results_dfs.append(process_file(file_path, args))
                print(f"\t...Processed {file_path}.")
        all_results_df = pd.concat(all_results_dfs)
        all_results_df.to_csv(args.output_file, index=False)
        print(f"Results saved to {args.output_file}")
        print("Summary of results (mean):")
        print(
            all_results_df.groupby(
                [
                    "model_name",
                    "prompt_template_name",
                    "context_length",
                    "generation_length",
                ]
            )[args.metrics].mean()
        )
        print("Summary of results (median):")
        print(
            all_results_df.groupby(
                [
                    "model_name",
                    "prompt_template_name",
                    "context_length",
                    "generation_length",
                ]
            )[args.metrics].median()
        )

    else:
        # Otherwise if path given is a single .csv file, just process it
        results_df = process_file(args.path_to_model_responses, args)
        gaps=[]
        for metric in args.metrics:
            print(f"Bootstrapping results for {metric}...")
            lower_bound, upper_bound, total_mean = boostrap_results(results_df, metric)
            print(f"Mean {metric}: {total_mean}")
            gap = (upper_bound - lower_bound) / 2
            print(f"Confidence interval gap: {gap:.3f}")
            print(f"Lower bound: {lower_bound:.3f}")
            print(f"Upper bound: {upper_bound:.3f}")
            gaps.append(gap)
            print()
        grouped_results=results_df.groupby(
        [
            "model_name",
            "prompt_template_name",
            # "context_length",
            "generation_length",
        ])[args.metrics].mean()
        gaps_df = pd.DataFrame([gaps], columns=args.metrics)
        
        # Append the gaps DataFrame to the results DataFrame
        results_df = pd.concat([grouped_results, gaps_df], ignore_index=True)
        results_df.to_csv(args.output_file, index=False)
        print(f"Processed {args.path_to_model_responses}")
        print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
