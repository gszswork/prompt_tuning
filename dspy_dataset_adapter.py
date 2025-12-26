"""
DSpy Dataset Adapter for Custom Datasets

This module adapts custom datasets loaded via load_dataset.py to DSpy's expected format.
It provides configurable train/valid/test splits and a metric function for evaluation.

Usage:
    from dspy_dataset_adapter import load_dspy_dataset

    dataset = load_dspy_dataset(
        dataset_name="CommonsenseQA",
        train_size=30,
        val_size=10
    )

    print(len(dataset.train), len(dataset.dev), len(dataset.test))

    # Use with DSpy
    module = dspy.ChainOfThought("question -> answer")
    evaluate = dspy.Evaluate(devset=dataset.dev, metric=dataset.metric)
"""

import dspy
from load_dataset import split_train_val_test


class CustomDSpyDataset:
    """
    A dataset class that wraps custom datasets in DSpy format.

    Attributes:
        train: List of dspy.Example objects for training
        dev: List of dspy.Example objects for validation/development
        test: List of dspy.Example objects for testing
        metric: Function to evaluate predictions against ground truth
    """

    def __init__(self, train_examples, dev_examples, test_examples, dataset_name):
        """
        Initialize the dataset with train/dev/test splits.

        Args:
            train_examples: List of dspy.Example objects for training
            dev_examples: List of dspy.Example objects for validation
            test_examples: List of dspy.Example objects for testing
            dataset_name: Name of the dataset (for metric customization)
        """
        self.train = train_examples
        self.dev = dev_examples
        self.test = test_examples
        self.dataset_name = dataset_name

    def metric(self, example, pred, trace=None):
        """
        Evaluation metric that compares prediction with ground truth.

        Args:
            example: dspy.Example object containing the ground truth
            pred: Prediction object with an 'answer' attribute
            trace: Optional trace information (unused)

        Returns:
            float: 1.0 if prediction matches ground truth, 0.0 otherwise
        """
        # Extract the prediction answer
        predicted_answer = pred.answer if hasattr(pred, 'answer') else str(pred)
        ground_truth = example.answer

        # Normalize strings for comparison
        predicted_answer = str(predicted_answer).strip().lower()
        ground_truth = str(ground_truth).strip().lower()

        # For numeric answers, try to compare as numbers
        try:
            pred_num = float(predicted_answer.replace(',', ''))
            gt_num = float(ground_truth.replace(',', ''))
            return 1.0 if abs(pred_num - gt_num) < 1e-6 else 0.0
        except (ValueError, AttributeError):
            pass

        # Check exact match first
        if predicted_answer == ground_truth:
            return 1.0

        # For single character answers (multiple choice like A, B, C, D)
        # Be strict - only match if it's at a word boundary or the exact string
        if len(ground_truth) <= 2:
            # Extract first non-whitespace character from prediction
            pred_first_char = predicted_answer.strip()[0] if predicted_answer.strip() else ""
            if pred_first_char == ground_truth:
                return 1.0
            # Also check if prediction starts with the answer (e.g., "A: complete job")
            if predicted_answer.startswith(ground_truth + ":") or predicted_answer.startswith(ground_truth + " "):
                return 1.0
            return 0.0

        # For longer answers, use containment matching
        # Check if ground truth is contained in prediction (for verbose answers)
        if ground_truth in predicted_answer:
            return 1.0

        # Check if prediction is contained in ground truth (for abbreviated answers)
        if predicted_answer in ground_truth:
            return 1.0

        return 0.0


def convert_to_dspy_examples(data_list):
    """
    Convert a list of {'x': input, 'y': target} dictionaries to dspy.Example objects.

    Args:
        data_list: List of dictionaries with 'x' (input) and 'y' (target) keys

    Returns:
        List of dspy.Example objects with 'question' and 'answer' fields
    """
    examples = []
    for item in data_list:
        # Create a DSpy Example with question and answer fields
        # DSpy expects named fields, commonly 'question' and 'answer'
        example = dspy.Example(
            question=str(item['x']),
            answer=str(item['y'])
        ).with_inputs("question")  # Mark 'question' as input field
        examples.append(example)
    return examples


def load_dspy_dataset(dataset_name, train_size, val_size):
    """
    Load a custom dataset and convert it to DSpy format with configurable splits.

    Args:
        dataset_name: Name of the dataset (e.g., 'CommonsenseQA', 'MultiArith')
        train_size: Number of samples for training set
        val_size: Number of samples for validation set

    Returns:
        CustomDSpyDataset object with train, dev, test attributes and metric function

    Example:
        >>> dataset = load_dspy_dataset("CommonsenseQA", train_size=30, val_size=10)
        >>> print(f"Train: {len(dataset.train)}, Dev: {len(dataset.dev)}, Test: {len(dataset.test)}")
        >>>
        >>> # Use with DSpy
        >>> module = dspy.ChainOfThought("question -> answer")
        >>> evaluate = dspy.Evaluate(devset=dataset.dev, metric=dataset.metric)
    """
    # Load data using existing function
    train_data, val_data, test_data = split_train_val_test(
        dataset_name=dataset_name,
        train_size=train_size,
        val_size=val_size
    )

    # Convert to DSpy Example format
    train_examples = convert_to_dspy_examples(train_data)
    dev_examples = convert_to_dspy_examples(val_data)
    test_examples = convert_to_dspy_examples(test_data)

    # Create and return the dataset object
    dataset = CustomDSpyDataset(
        train_examples=train_examples,
        dev_examples=dev_examples,
        test_examples=test_examples,
        dataset_name=dataset_name
    )

    return dataset


if __name__ == "__main__":
    # Example usage and testing
    print("Testing DSpy Dataset Adapter")
    print("-" * 60)

    # Test with CommonsenseQA
    print("\nLoading CommonsenseQA dataset...")
    dataset = load_dspy_dataset("CommonsenseQA", train_size=30, val_size=10)

    print(f"Train set size: {len(dataset.train)}")
    print(f"Dev set size: {len(dataset.dev)}")
    print(f"Test set size: {len(dataset.test)}")

    # Show a sample
    if len(dataset.train) > 0:
        print("\nSample training example:")
        sample = dataset.train[0]
        print(f"  Question: {sample.question[:100]}...")
        print(f"  Answer: {sample.answer}")

    # Test metric function
    print("\nTesting metric function:")
    if len(dataset.train) > 0:
        sample = dataset.train[0]

        # Create a mock prediction object
        class MockPred:
            def __init__(self, answer):
                self.answer = answer

        # Test exact match
        pred_correct = MockPred(sample.answer)
        score_correct = dataset.metric(sample, pred_correct)
        print(f"  Correct prediction score: {score_correct}")

        # Test incorrect match
        pred_incorrect = MockPred("wrong answer")
        score_incorrect = dataset.metric(sample, pred_incorrect)
        print(f"  Incorrect prediction score: {score_incorrect}")

    print("\n" + "=" * 60)
    print("Dataset adapter is ready to use with DSpy!")
    print("=" * 60)
