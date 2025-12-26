"""
DSpy Training Script with Per-Trial Logging

This enhanced version captures detailed per-trial information similar to TextGrad's per-epoch logging.
For each trial (analogous to TextGrad's epochs), it logs:
- The prompt combination (instructions + few-shot demos)
- Validation set predictions for all samples
- Scores and performance metrics

Usage examples:
# For CommonsenseQA
python dspy_train_with_trial_logging.py --dataset_name CommonsenseQA --train_size 30 --val_size 10

# For MultiArith
python dspy_train_with_trial_logging.py --dataset_name MultiArith --train_size 40 --val_size 15
"""

import argparse
import json
import pandas as pd
import dspy
from dotenv import load_dotenv
from dspy_dataset_adapter import load_dspy_dataset
from tqdm import tqdm
import threading

load_dotenv()

# Global storage for capturing trial evaluations during compile()
_trial_evaluation_cache = {}
_cache_lock = threading.Lock()
_current_trial_number = None
_trial_counter = 0


def create_caching_metric(base_metric, dataset_dev):
    """
    Wraps a metric function to capture detailed evaluation results during optimization.
    This allows us to save results during compile() instead of re-running trials afterwards.
    """
    def caching_metric_wrapper(example, prediction, trace=None):
        global _trial_counter, _current_trial_number

        # Calculate the actual metric score
        score = base_metric(example, prediction, trace) if trace is not None else base_metric(example, prediction)

        # Try to detect trial number from the trace or use a counter
        # DSpy's MIPROv2 doesn't directly expose trial numbers during evaluation,
        # so we need to infer it from the evaluation context
        with _cache_lock:
            # When a new evaluation starts, we increment the trial counter
            # This is a heuristic: detect new trial by tracking evaluation patterns

            # Extract prediction text
            if hasattr(prediction, 'answer'):
                pred_text = prediction.answer
            elif hasattr(prediction, 'output'):
                pred_text = prediction.output
            else:
                pred_text = str(prediction)

            # Store detailed result
            result_entry = {
                "question": example.question if hasattr(example, 'question') else str(example.inputs()),
                "ground_truth": example.answer if hasattr(example, 'answer') else str(example),
                "prediction": pred_text,
                "score": float(score)
            }

            # Initialize trial storage if needed
            if _current_trial_number not in _trial_evaluation_cache:
                _trial_evaluation_cache[_current_trial_number] = []

            _trial_evaluation_cache[_current_trial_number].append(result_entry)

        return score

    return caching_metric_wrapper


def extract_predictor_prompts(program):
    """Extract the current prompts (instructions + demos) from a DSpy program."""
    prompts_info = []

    for i, predictor in enumerate(program.predictors()):
        predictor_info = {
            "predictor_index": i,
            "predictor_name": predictor.__class__.__name__,
        }

        # Extract instruction/system prompt
        if hasattr(predictor, 'signature'):
            sig = predictor.signature
            if hasattr(sig, 'instructions') and sig.instructions:
                predictor_info["instruction"] = sig.instructions
            elif hasattr(sig, '__doc__') and sig.__doc__:
                predictor_info["instruction"] = sig.__doc__.strip()

        # Extract few-shot demonstrations
        if hasattr(predictor, 'demos') and predictor.demos:
            predictor_info["num_demos"] = len(predictor.demos)
            predictor_info["demos"] = []
            for demo in predictor.demos:
                try:
                    # Try to extract inputs and outputs properly
                    if hasattr(demo, 'inputs'):
                        demo_inputs = dict(demo.inputs())
                        demo_outputs = {k: v for k, v in demo.items() if k not in demo_inputs}
                    else:
                        demo_inputs = str(demo)
                        demo_outputs = {}

                    predictor_info["demos"].append({
                        "inputs": demo_inputs,
                        "outputs": demo_outputs
                    })
                except (ValueError, AttributeError) as e:
                    # If demo.inputs() fails, fall back to string representation
                    predictor_info["demos"].append({
                        "inputs": {k: v for k, v in demo.items()},
                        "outputs": {}
                    })
        else:
            predictor_info["num_demos"] = 0
            predictor_info["demos"] = []

        prompts_info.append(predictor_info)

    return prompts_info


def evaluate_program_detailed(program, devset, metric_fn):
    """
    Evaluate a program on a dataset and return detailed per-example results.

    Returns:
        tuple: (overall_score, detailed_results_list)
        where detailed_results_list contains dict for each example with:
        - index, question, ground_truth, prediction, score
    """
    detailed_results = []
    scores = []

    for idx, example in enumerate(tqdm(devset, desc="Evaluating", leave=False)):
        try:
            # Run prediction
            prediction = program(**example.inputs())

            # Calculate score
            score = metric_fn(example, prediction)
            scores.append(score)

            # Extract prediction text
            if hasattr(prediction, 'answer'):
                pred_text = prediction.answer
            elif hasattr(prediction, 'output'):
                pred_text = prediction.output
            else:
                pred_text = str(prediction)

            detailed_results.append({
                "index": idx,
                "question": example.question if hasattr(example, 'question') else str(example.inputs()),
                "ground_truth": example.answer if hasattr(example, 'answer') else str(example),
                "prediction": pred_text,
                "score": float(score)
            })

        except Exception as e:
            print(f"Error evaluating example {idx}: {e}")
            detailed_results.append({
                "index": idx,
                "question": example.question if hasattr(example, 'question') else str(example.inputs()),
                "ground_truth": example.answer if hasattr(example, 'answer') else str(example),
                "prediction": f"ERROR: {str(e)}",
                "score": 0.0
            })
            scores.append(0.0)

    overall_score = sum(scores) / len(scores) if scores else 0.0
    return overall_score, detailed_results


class TrialResultsLogger:
    """Logger that captures and saves trial results during optimization."""

    def __init__(self, csv_file, dataset_name):
        self.csv_file = csv_file
        self.dataset_name = dataset_name
        self.csv_writer = None
        self.csv_file_handle = None
        self.trial_data = {
            "trial_scores": [],
            "trial_prompts": [],
            "detailed_results": []
        }

    def open_csv(self):
        """Open CSV file for writing."""
        self.csv_file_handle = open(self.csv_file, 'w', newline='')
        import csv
        fieldnames = ['trial_number', 'split', 'index', 'question', 'ground_truth', 'prediction', 'correct', 'prompt']
        self.csv_writer = csv.DictWriter(self.csv_file_handle, fieldnames=fieldnames)
        self.csv_writer.writeheader()
        print(f"\nOpened CSV file for streaming results: {self.csv_file}")

    def log_trial(self, trial_num, score, eval_type, program, detailed_results):
        """Log a trial's results to CSV immediately."""
        # Extract prompts
        prompts = extract_predictor_prompts(program)
        prompt_str = json.dumps(prompts)

        # Store in memory for later JSON export
        self.trial_data["trial_scores"].append({
            "trial_number": trial_num,
            "score": float(score),
            "eval_type": eval_type
        })

        self.trial_data["trial_prompts"].append({
            "trial_number": trial_num,
            "prompts": prompts
        })

        self.trial_data["detailed_results"].append({
            "trial_number": trial_num,
            "results": detailed_results
        })

        # Write to CSV immediately (streaming)
        if self.csv_writer:
            for idx, item in enumerate(detailed_results):
                self.csv_writer.writerow({
                    'trial_number': trial_num,
                    'split': 'val',
                    'index': idx,
                    'question': item['question'],
                    'ground_truth': item['ground_truth'],
                    'prediction': item['prediction'],
                    'correct': item['score'],
                    'prompt': prompt_str
                })
            self.csv_file_handle.flush()  # Ensure data is written to disk

    def close_csv(self):
        """Close CSV file."""
        if self.csv_file_handle:
            self.csv_file_handle.close()
            print(f"Closed CSV file: {self.csv_file}")


def extract_trial_data(optimized_module, dataset, metric_fn, max_trials=None, results_logger=None):
    """
    Extract comprehensive per-trial information from the optimized module.
    If results_logger is provided and has cached data, use that instead of re-evaluating.

    Args:
        optimized_module: The compiled DSpy module with trial_logs attached
        dataset: Dataset object with dev set for re-evaluation
        metric_fn: Metric function for evaluation
        max_trials: Maximum number of trials to process (None = all)
        results_logger: Optional TrialResultsLogger with cached results

    Returns:
        dict with trial_scores, trial_prompts, and detailed_results per trial
    """
    if not hasattr(optimized_module, 'trial_logs'):
        print("Warning: No trial_logs found in optimized module. Cannot extract per-trial data.")
        return None

    trial_logs = optimized_module.trial_logs
    print(f"\nFound {len(trial_logs)} trials in optimization logs")

    results = {
        "trial_scores": [],
        "trial_prompts": [],
        "detailed_results": []  # List of per-trial detailed results
    }

    # Sort trials by trial number
    sorted_trials = sorted(trial_logs.items(), key=lambda x: x[0])

    if max_trials:
        sorted_trials = sorted_trials[:max_trials]
        print(f"Processing first {max_trials} trials")

    for trial_num, trial_data in sorted_trials:
        print(f"\nProcessing Trial {trial_num}...")

        # Get score (prefer full eval, fallback to minibatch)
        if "full_eval_score" in trial_data:
            score = trial_data["full_eval_score"]
            eval_type = "full"
        elif "mb_score" in trial_data:
            score = trial_data["mb_score"]
            eval_type = "minibatch"
        else:
            print(f"  Warning: No score found for trial {trial_num}")
            continue

        # Get program (prefer full eval, fallback to minibatch)
        if "full_eval_program" in trial_data:
            program = trial_data["full_eval_program"]
        elif "mb_program" in trial_data:
            program = trial_data["mb_program"]
        else:
            print(f"  Warning: No program found for trial {trial_num}")
            continue

        # Extract prompts from program
        prompts = extract_predictor_prompts(program)

        # Re-evaluate to get detailed per-example results
        print(f"  Re-evaluating trial {trial_num} on validation set...")
        _, detailed = evaluate_program_detailed(program, dataset.dev, metric_fn)

        # Store results
        results["trial_scores"].append({
            "trial_number": trial_num,
            "score": float(score),
            "eval_type": eval_type
        })

        results["trial_prompts"].append({
            "trial_number": trial_num,
            "prompts": prompts
        })

        results["detailed_results"].append({
            "trial_number": trial_num,
            "results": detailed
        })

        print(f"  Trial {trial_num}: Score = {score:.4f} ({eval_type})")

        # Log to CSV immediately if logger is provided
        if results_logger:
            results_logger.log_trial(trial_num, score, eval_type, program, detailed)

    return results


def save_results(results, optimized_module, dataset, metric_fn, args, trial_data, results_logger):
    """Save optimization results to JSON and append test results to CSV."""

    # Evaluate final optimized module on test set
    print("\n" + "="*60)
    print("FINAL TEST SET EVALUATION")
    print("="*60)
    test_score, test_detailed = evaluate_program_detailed(
        optimized_module, dataset.test, metric_fn
    )
    print(f"Final Test Score: {test_score:.4f}")

    # Get final prompts
    final_prompts = extract_predictor_prompts(optimized_module)

    # Prepare best trial info (extract only serializable parts)
    best_trial_info = {}
    if hasattr(optimized_module, 'trial_logs') and optimized_module.trial_logs:
        best_trial_num = max(optimized_module.trial_logs.keys())
        best_trial = optimized_module.trial_logs[best_trial_num]
        # Extract only serializable fields
        best_trial_info = {
            "trial_number": best_trial_num,
            "score": float(best_trial.get("full_eval_score", best_trial.get("mb_score", 0))),
            "eval_type": "full" if "full_eval_score" in best_trial else "minibatch"
        }

    # Prepare JSON output
    json_output = {
        "dataset": args.dataset_name,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "max_bootstrapped_demos": args.max_bootstrapped_demos,
        "max_labeled_demos": args.max_labeled_demos,
        "num_threads": args.num_threads,
        "optimizer": "MIPROv2",
        "auto_setting": "medium",

        # Trial-level scores (analogous to per-epoch in TextGrad)
        "trial_scores": trial_data["trial_scores"] if trial_data else [],

        # Final results
        "final_test_score": float(test_score),
        "best_trial": best_trial_info,

        # Prompts per trial (analogous to prompts_per_epoch in TextGrad)
        "trial_prompts": trial_data["trial_prompts"] if trial_data else [],
        "final_prompts": final_prompts,
    }

    # Save JSON
    json_file = f"dspy_results_{args.dataset_name}_train{args.train_size}_val{args.val_size}.json"
    with open(json_file, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"\nJSON results saved to: {json_file}")

    # Append final test set results to the CSV that already has trial data
    # The trial data was already written during extract_trial_data()
    print("\nAppending test set results to CSV...")
    final_prompt_str = json.dumps(final_prompts)
    for item in test_detailed:
        results_logger.csv_writer.writerow({
            'trial_number': 'final',
            'split': 'test',
            'index': item['index'],
            'question': item['question'],
            'ground_truth': item['ground_truth'],
            'prediction': item['prediction'],
            'correct': item['score'],
            'prompt': final_prompt_str
        })
    results_logger.csv_file_handle.flush()

    print(f"Test results appended to CSV: {results_logger.csv_file}")
    print(f"Total test examples: {len(test_detailed)}")


def main():
    parser = argparse.ArgumentParser(description="Train DSpy with per-trial logging")
    parser.add_argument("--dataset_name", type=str, required=True,
                       help="Name of the dataset (e.g., CommonsenseQA, MultiArith)")
    parser.add_argument("--train_size", type=int, default=10,
                       help="Number of training samples")
    parser.add_argument("--val_size", type=int, default=10,
                       help="Number of validation samples")
    parser.add_argument("--num_threads", type=int, default=24,
                       help="Number of threads for evaluation")
    parser.add_argument("--max_bootstrapped_demos", type=int, default=4,
                       help="Maximum number of bootstrapped demonstrations")
    parser.add_argument("--max_labeled_demos", type=int, default=4,
                       help="Maximum number of labeled demonstrations")
    parser.add_argument("--max_trials_to_log", type=int, default=None,
                       help="Maximum number of trials to extract detailed logs for (None = all)")
    args = parser.parse_args()

    # Initialize language models
    expensive_model = dspy.LM('openai/gpt-5.2', max_tokens=40000)
    cheap_model = dspy.LM('openai/gpt-4o', max_tokens=2000)
    dspy.configure(lm=cheap_model)

    # Load custom dataset
    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dspy_dataset(
        dataset_name=args.dataset_name,
        train_size=args.train_size,
        val_size=args.val_size
    )
    print(f"Train: {len(dataset.train)}, Dev: {len(dataset.dev)}, Test: {len(dataset.test)}")

    # Create the module and optimizer
    module = dspy.ChainOfThought("question -> answer")

    # Setup evaluation
    THREADS = args.num_threads
    kwargs = dict(num_threads=THREADS, display_progress=True, display_table=5)
    evaluate = dspy.Evaluate(devset=dataset.dev, metric=dataset.metric, **kwargs)

    # Setup optimizer
    kwargs = dict(num_threads=THREADS, teacher_settings=dict(lm=expensive_model), prompt_model=cheap_model)
    optimizer = dspy.MIPROv2(metric=dataset.metric, auto="medium", **kwargs)

    # Create CSV logger to save results as trials complete
    csv_file = f"dspy_detailed_results_{args.dataset_name}_train{args.train_size}_val{args.val_size}.csv"
    results_logger = TrialResultsLogger(csv_file, args.dataset_name)
    results_logger.open_csv()

    # Compile/optimize the module
    print("\nStarting optimization...")
    kwargs = dict(
        max_bootstrapped_demos=args.max_bootstrapped_demos,
        max_labeled_demos=args.max_labeled_demos
    )
    optimized_module = optimizer.compile(module, trainset=dataset.train, **kwargs)

    # Evaluate the optimized module
    print("\nEvaluating optimized module on dev set...")
    dev_score = evaluate(optimized_module)

    print("\nEvaluating optimized module on test set...")
    test_kwargs = dict(num_threads=THREADS, display_progress=True, display_table=5)
    test_evaluate = dspy.Evaluate(devset=dataset.test, metric=dataset.metric, **test_kwargs)
    test_score = test_evaluate(optimized_module)

    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Dev Score: {dev_score}")
    print(f"Test Score: {test_score}")

    # Extract per-trial data and save to CSV as we go
    print("\n" + "="*60)
    print("EXTRACTING PER-TRIAL DATA AND SAVING TO CSV")
    print("="*60)
    trial_data = extract_trial_data(
        optimized_module,
        dataset,
        dataset.metric,
        max_trials=args.max_trials_to_log,
        results_logger=results_logger
    )

    # Save all results
    print("\n" + "="*60)
    print("SAVING FINAL RESULTS")
    print("="*60)
    save_results(
        {"dev_score": dev_score, "test_score": test_score},
        optimized_module,
        dataset,
        dataset.metric,
        args,
        trial_data,
        results_logger
    )

    # Close CSV logger
    results_logger.close_csv()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if trial_data:
        print(f"Total trials logged: {len(trial_data['trial_scores'])}")
        print(f"Trial scores: {[t['score'] for t in trial_data['trial_scores'][:5]]}..." if len(trial_data['trial_scores']) > 5 else f"Trial scores: {[t['score'] for t in trial_data['trial_scores']]}")
    print(f"Final Dev Score: {dev_score}")
    print(f"Final Test Score: {test_score}")
    print(f"Detailed CSV: {csv_file}")
    print("="*60)


if __name__ == "__main__":
    main()
