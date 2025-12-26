"""
DSpy Training Script with Custom Dataset

Usage examples:
# For CommonsenseQA
python dspy_train.py --dataset_name CommonsenseQA --train_size 30 --val_size 10

# For MultiArith
python dspy_train.py --dataset_name MultiArith --train_size 40 --val_size 15
"""

import argparse
import dspy
from dotenv import load_dotenv
from dspy_dataset_adapter import load_dspy_dataset

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Train DSpy on custom dataset")
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
    args = parser.parse_args()

    # Initialize language models
    # Expensive model 
    expensive_model = dspy.LM('openai/gpt-5.2', max_tokens=40000)
    # Cheap model
    cheap_model = dspy.LM('openai/gpt-4o', max_tokens=2000)

    dspy.configure(lm=cheap_model)  # we'll use gpt-4o-mini as the default LM, unless otherwise specified

    # Load custom dataset with configurable splits
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
    test_evaluate = dspy.Evaluate(devset=dataset.test, metric=dataset.metric, **kwargs)
    test_score = test_evaluate(optimized_module)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Dev Score: {dev_score}")
    print(f"Test Score: {test_score}")
    print("="*60)


if __name__ == "__main__":
    main()
