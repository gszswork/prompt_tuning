"""
# For CommonsenseQA
python textgrad_train.py --dataset_name CommonsenseQA --train_size 30 --val_size 10 --epochs 2

# For MultiArith
python textgrad_train.py --dataset_name MultiArith --train_size 40 --val_size 15 --batch_size 5
"""

import argparse
import concurrent
import json
import os
import random

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import textgrad as tg
from tqdm import tqdm

from load_dataset import split_train_val_test

load_dotenv(override=True)


def append_to_csv(csv_file, rows):
    """Append rows to CSV file incrementally"""
    if not rows:
        return
    
    df_new = pd.DataFrame(rows)
    
    # Check if file exists
    if os.path.exists(csv_file):
        # Append mode - read existing and append
        df_existing = pd.read_csv(csv_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(csv_file, index=False)
    else:
        # First write - create new file
        df_new.to_csv(csv_file, index=False)


def set_seed(seed):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)


def eval_sample(item, eval_fn, model):
    x, y = item
    x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
    y = tg.Variable(y, requires_grad=False, role_description="correct answer for the query")
    response = model(x)

    # Format the evaluation input as a single text
    eval_input_text = f"Prediction: {response.value}\nGround Truth: {y.value}"
    eval_input = tg.Variable(eval_input_text, requires_grad=False, role_description="evaluation input")

    eval_output_variable = eval_fn(eval_input)
    return int(eval_output_variable.value.strip())


def eval_dataset(test_set, eval_fn, model, max_samples=None, return_details=False):
    if max_samples is None:
        max_samples = len(test_set)

    accuracy_list = []
    details_list = []  # Store question, prediction, ground truth, correctness

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for idx, sample in enumerate(test_set):
            future = executor.submit(eval_sample, sample, eval_fn, model)
            futures.append((future, idx, sample))
            if len(futures) >= max_samples:
                break

        tqdm_loader = tqdm(concurrent.futures.as_completed([f[0] for f in futures]), total=len(futures), position=0)
        future_map = {f[0]: (f[1], f[2]) for f in futures}

        for future in tqdm_loader:
            acc_item = future.result()
            accuracy_list.append(acc_item)
            tqdm_loader.set_description(f"Accuracy: {np.mean(accuracy_list):.4f}")

            if return_details:
                idx, sample = future_map[future]
                question, ground_truth = sample
                details_list.append({
                    'index': idx,
                    'question': question,
                    'ground_truth': ground_truth,
                    'correct': acc_item
                })

    if return_details:
        return accuracy_list, details_list
    return accuracy_list


def run_validation_revert(system_prompt, previous_val_performance, previous_prompt, previous_epoch, model, eval_fn, val_set, current_epoch):
    """Validate and potentially revert prompt if performance drops"""
    # Get detailed validation results for the current prompt
    val_acc, val_details = eval_dataset(val_set, eval_fn, model, return_details=True)
    val_performance = np.mean(val_acc)

    print(f"Current validation performance: {val_performance:.4f}")
    print(f"Previous validation performance: {previous_val_performance:.4f}")

    current_prompt = system_prompt.get_value()
    
    if val_performance < previous_val_performance:
        print(f"Performance dropped! Reverting prompt.")
        print(f"Rejected prompt: {current_prompt}")
        system_prompt.set_value(previous_prompt)
        return previous_val_performance, previous_prompt, previous_epoch, val_details, current_prompt
    else:
        print("Performance improved or maintained. Keeping new prompt.")
        return val_performance, current_prompt, current_epoch, val_details, current_prompt


def create_eval_function(llm_api):
    eval_system_prompt = """You are an evaluator assessing if a prediction matches the ground truth answer.

Your task:
1. Compare the prediction with the ground truth answer
2. Determine if they are semantically equivalent (even if worded differently)
3. For numerical answers, check if the values match
4. For multiple choice, check if the selected option is correct

Respond with exactly "1" if the prediction is correct, or "0" if incorrect.
Do not provide any explanation, just output the single digit."""

    eval_fn = tg.BlackboxLLM(llm_api, tg.Variable(eval_system_prompt, requires_grad=False, role_description="evaluation system prompt for assessing predictions"))
    return eval_fn


def convert_dataset_format(dataset):
    return [(item['x'], str(item['y'])) for item in dataset]


def get_task_description(dataset_name):
    prompts = {
        "CommonsenseQA": "Let's think step by step.",
        "MultiArith": "Solve the following math word problem. Show your reasoning and provide the final numerical answer.",
        "default": "Answer the following question accurately and concisely."
    }
    return prompts.get(dataset_name, prompts["default"])


def create_data_loader(dataset, batch_size, shuffle=True):
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)

    for i in range(0, len(dataset), batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_x = [dataset[idx][0] for idx in batch_indices]
        batch_y = [dataset[idx][1] for idx in batch_indices]
        yield batch_x, batch_y


def main():
    parser = argparse.ArgumentParser(description="Train TextGrad on custom dataset")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset (e.g., CommonsenseQA, MultiArith)")
    parser.add_argument("--train_size", type=int, default=50, help="Number of training samples")
    parser.add_argument("--val_size", type=int, default=20, help="Number of validation samples")
    parser.add_argument("--batch_size", type=int, default=3, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=3, help="Number of optimization steps per epoch")
    parser.add_argument("--eval_engine", type=str, default="experimental:gpt-5.2", help="Engine for evaluation and optimization")
    parser.add_argument("--test_engine", type=str, default="gpt-4o", help="Engine to optimize (test model)")
    parser.add_argument("--seed", type=int, default=12, help="Random seed")
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Load dataset
    print(f"Loading dataset: {args.dataset_name}")
    train_data, val_data, test_data = split_train_val_test(
        args.dataset_name,
        args.train_size,
        args.val_size
    )

    # Convert to TextGrad format (list of tuples)
    train_set = convert_dataset_format(train_data)
    val_set = convert_dataset_format(val_data)
    test_set = convert_dataset_format(test_data)

    print(f"Train/Val/Test Set Lengths: {len(train_set)}, {len(val_set)}, {len(test_set)}")

    # Initialize LLM engines
    print(f"Initializing engines: eval={args.eval_engine}, test={args.test_engine}")
    llm_api_eval = tg.get_engine(engine_name=args.eval_engine)
    llm_api_test = tg.get_engine(engine_name=args.test_engine)
    tg.set_backward_engine(llm_api_eval, override=True)

    # Create evaluation function
    eval_fn = create_eval_function(llm_api_eval)

    # Get initial system prompt
    STARTING_SYSTEM_PROMPT = get_task_description(args.dataset_name)
    print(f"\nStarting system prompt:\n{STARTING_SYSTEM_PROMPT}\n")

    # Create TextGrad variables and model
    system_prompt = tg.Variable(
        STARTING_SYSTEM_PROMPT,
        requires_grad=True,
        role_description="structured system prompt to a language model that specifies the behavior and strategies for the task"
    )
    model = tg.BlackboxLLM(llm_api_test, system_prompt)

    # Create optimizer
    optimizer = tg.TextualGradientDescent(engine=llm_api_eval, parameters=[system_prompt])

    # Create data loader
    train_loader = create_data_loader(train_set, batch_size=args.batch_size, shuffle=True)

    # Initialize CSV file path early for incremental logging
    csv_dir = "results/TextGrad"
    os.makedirs(csv_dir, exist_ok=True)
    csv_file = f"{csv_dir}/TextGrad_results_{args.dataset_name}_train{args.train_size}_val{args.val_size}_epochs{args.epochs}_batch{args.batch_size}_steps{args.steps_per_epoch}_seed{args.seed}.csv"
    
    # Initialize CSV file with headers if it doesn't exist
    if not os.path.exists(csv_file):
        df_header = pd.DataFrame(columns=['epoch', 'step', 'split', 'index', 'question', 'ground_truth', 'correct', 'prompt', 'accepted'])
        df_header.to_csv(csv_file, index=False)

    # Initialize results tracking
    results = {
        "train_acc_per_epoch": [],
        "val_acc_per_epoch": [],
        "prompts_per_epoch": [],
        "detailed_results": [],  # Will store per-question correctness (train and val only during training)
        "all_attempts": []  # Will store all validation attempts (each step) with their prompts and results
    }

    # Evaluate initial performance (Epoch 0)
    print("="*60)
    print("EPOCH 0 (Initial Performance)")
    print("="*60)

    print("\nTrain set evaluation:")
    initial_train_acc, train_details = eval_dataset(train_set, eval_fn, model, return_details=True)

    print("Validation set evaluation:")
    initial_val_acc, val_details = eval_dataset(val_set, eval_fn, model, return_details=True)

    # Store initial results
    results["train_acc_per_epoch"].append(float(np.mean(initial_train_acc)))
    results["val_acc_per_epoch"].append(float(np.mean(initial_val_acc)))
    results["prompts_per_epoch"].append(system_prompt.get_value())

    # Store detailed results
    results["detailed_results"].append({
        "epoch": 0,
        "train": train_details,
        "val": val_details
    })

    # Write epoch 0 results to CSV immediately
    epoch_0_prompt = results['prompts_per_epoch'][0]
    csv_rows_epoch0 = []
    for item in train_details:
        csv_rows_epoch0.append({
            'epoch': 0,
            'step': 0,
            'split': 'train',
            'index': item['index'],
            'question': item['question'],
            'ground_truth': item['ground_truth'],
            'correct': item['correct'],
            'prompt': epoch_0_prompt,
            'accepted': True
        })
    for item in val_details:
        csv_rows_epoch0.append({
            'epoch': 0,
            'step': 0,
            'split': 'val',
            'index': item['index'],
            'question': item['question'],
            'ground_truth': item['ground_truth'],
            'correct': item['correct'],
            'prompt': epoch_0_prompt,
            'accepted': True
        })
    append_to_csv(csv_file, csv_rows_epoch0)
    print(f"Epoch 0 results written to CSV: {len(csv_rows_epoch0)} rows")

    print(f"\nEpoch 0 Performance:")
    print(f"  Train Accuracy: {np.mean(initial_train_acc):.4f}")
    print(f"  Validation Accuracy: {np.mean(initial_val_acc):.4f}\n")

    # Track best validation performance for reverting and final test evaluation
    best_val_performance = float(np.mean(initial_val_acc))
    best_prompt = system_prompt.get_value()
    best_epoch = 0

    # Training loop
    print("Starting training...")
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")

        # Recreate data loader for each epoch
        train_loader = create_data_loader(train_set, batch_size=args.batch_size, shuffle=True)

        for steps, (batch_x, batch_y) in enumerate((pbar := tqdm(train_loader, position=0))):
            pbar.set_description(f"Training step {steps + 1}. Epoch {epoch + 1}")

            # Zero gradients
            optimizer.zero_grad()

            # Compute loss for batch
            losses = []
            for (x, y) in zip(batch_x, batch_y):
                x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
                y = tg.Variable(y, requires_grad=False, role_description="correct answer for the query")
                response = model(x)

                # Format the evaluation input as a single text
                eval_input_text = f"Prediction: {response.value}\nGround Truth: {y.value}"
                eval_input = tg.Variable(eval_input_text, requires_grad=False, role_description="evaluation input")
                eval_output_variable = eval_fn(eval_input)

                losses.append(eval_output_variable)

            # Backward pass
            total_loss = tg.sum(losses)
            total_loss.backward()

            # Update prompt
            optimizer.step()

            # Validate and potentially revert after each step
            print(f"\n--- Validation after step {steps + 1} ---")
            best_val_performance, best_prompt, best_epoch, val_details_attempt, attempted_prompt = run_validation_revert(
                system_prompt, best_val_performance, best_prompt, best_epoch, model, eval_fn, val_set, epoch + 1
            )
            
            # Store validation results for this attempt (even if it was rejected)
            # attempted_prompt is the draft enhanced prompt that was validated
            accepted = (attempted_prompt == best_prompt)  # True if this prompt was kept
            results["all_attempts"].append({
                "epoch": epoch + 1,
                "step": steps + 1,
                "prompt": attempted_prompt,
                "val_performance": float(np.mean([item['correct'] for item in val_details_attempt])),
                "val_details": val_details_attempt,
                "accepted": accepted
            })
            
            # Write validation attempt results to CSV immediately
            csv_rows_attempt = []
            for item in val_details_attempt:
                csv_rows_attempt.append({
                    'epoch': epoch + 1,
                    'step': steps + 1,
                    'split': 'val',
                    'index': item['index'],
                    'question': item['question'],
                    'ground_truth': item['ground_truth'],
                    'correct': item['correct'],
                    'prompt': attempted_prompt,
                    'accepted': accepted
                })
            append_to_csv(csv_file, csv_rows_attempt)
            print(f"Validation attempt results written to CSV: {len(csv_rows_attempt)} rows (accepted: {accepted})")

            if steps + 1 >= args.steps_per_epoch:
                break

        # Evaluate train and validation datasets after each epoch
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1} EVALUATION")
        print(f"{'='*60}")

        print("\nTrain set evaluation:")
        train_acc, train_details = eval_dataset(train_set, eval_fn, model, return_details=True)

        print("Validation set evaluation:")
        val_acc, val_details = eval_dataset(val_set, eval_fn, model, return_details=True)

        # Store epoch results
        results["train_acc_per_epoch"].append(float(np.mean(train_acc)))
        results["val_acc_per_epoch"].append(float(np.mean(val_acc)))
        results["prompts_per_epoch"].append(system_prompt.get_value())

        # Store detailed results
        results["detailed_results"].append({
            "epoch": epoch + 1,
            "train": train_details,
            "val": val_details
        })

        print(f"\nEpoch {epoch + 1} Performance:")
        print(f"  Train Accuracy: {np.mean(train_acc):.4f}")
        print(f"  Validation Accuracy: {np.mean(val_acc):.4f}")
        print(f"\nOptimized prompt:\n{system_prompt.get_value()}\n")

    # Final results - Evaluate on test set with best validated prompt
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")

    print(f"\nBest validation performance: {best_val_performance:.4f} (achieved at epoch {best_epoch})")
    print(f"\nBest validated prompt:\n{best_prompt}\n")

    # Set model to use best validated prompt
    system_prompt.set_value(best_prompt)

    print("="*60)
    print("FINAL TEST SET EVALUATION (using best validated prompt)")
    print("="*60)
    test_acc, test_details = eval_dataset(test_set, eval_fn, model, return_details=True)
    final_test_acc = float(np.mean(test_acc))

    print(f"\nFinal Test Accuracy: {final_test_acc:.4f}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\nInitial (Epoch 0):")
    print(f"  Train Accuracy: {results['train_acc_per_epoch'][0]:.4f}")
    print(f"  Val Accuracy: {results['val_acc_per_epoch'][0]:.4f}")
    print(f"\nBest Model (Epoch {best_epoch}):")
    print(f"  Val Accuracy: {best_val_performance:.4f}")
    print(f"  Test Accuracy: {final_test_acc:.4f}")

    # Save results to JSON with unique filename
    json_file = f"results_{args.dataset_name}_train{args.train_size}_val{args.val_size}_epochs{args.epochs}_batch{args.batch_size}_steps{args.steps_per_epoch}_seed{args.seed}.json"
    with open(json_file, 'w') as f:
        json.dump({
            "dataset": args.dataset_name,
            "train_size": args.train_size,
            "val_size": args.val_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "steps_per_epoch": args.steps_per_epoch,
            "seed": args.seed,
            "eval_engine": args.eval_engine,
            "test_engine": args.test_engine,
            "train_acc_per_epoch": results['train_acc_per_epoch'],
            "val_acc_per_epoch": results['val_acc_per_epoch'],
            "final_test_acc": final_test_acc,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_performance,
            "prompts_per_epoch": results['prompts_per_epoch'],
            "initial_prompt": results['prompts_per_epoch'][0],
            "best_prompt": best_prompt
        }, f, indent=2)

    print(f"\nJSON results saved to: {json_file}")

    # Write final test set results to CSV (all other results already written incrementally)
    csv_rows_test = []
    for item in test_details:
        csv_rows_test.append({
            'epoch': best_epoch,
            'step': 'final',
            'split': 'test',
            'index': item['index'],
            'question': item['question'],
            'ground_truth': item['ground_truth'],
            'correct': item['correct'],
            'prompt': best_prompt,
            'accepted': True
        })
    append_to_csv(csv_file, csv_rows_test)
    
    # Count total rows in CSV
    df_final = pd.read_csv(csv_file)
    total_rows = len(df_final)

    print(f"Final test results written to CSV: {len(csv_rows_test)} rows")
    print(f"Detailed CSV results saved to: {csv_file}")
    print(f"\nTotal rows in CSV: {total_rows}")


if __name__ == "__main__":
    main()
