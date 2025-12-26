import pandas as pd

def load_dataset(dataset_name: str):
    file_path = f"dataset/{dataset_name}.csv"
    df = pd.read_csv(file_path)

    if dataset_name == "CommonsenseQA":
        # input = input + input_choices, target = target
        input_list = (df['input'] + " " + df['input_choices']).tolist()
        target_list = df['target'].tolist()
    elif dataset_name == "MultiArith":
        # input = sQuestion, target = lSolutions
        input_list = df['sQuestion'].tolist()
        target_list = df['lSolutions'].tolist()
    else:
        # input = input, target = target
        input_list = df['input'].tolist()
        target_list = df['target'].tolist()

    return input_list, target_list

def split_train_val_test(dataset_name: str, train_size: int, val_size: int):
    input_list, target_list = load_dataset(dataset_name)

    # Assert for out-of-index
    assert train_size + val_size <= len(input_list), \
        f"train_size ({train_size}) + val_size ({val_size}) = {train_size + val_size} exceeds dataset size ({len(input_list)})"

    # Split into train, valid, test (non-overlapping)
    train = [{'x': x, 'y': y} for x, y in zip(input_list[:train_size], target_list[:train_size])]
    valid = [{'x': x, 'y': y} for x, y in zip(input_list[train_size:train_size + val_size], target_list[train_size:train_size + val_size])]
    test = [{'x': x, 'y': y} for x, y in zip(input_list[train_size + val_size:], target_list[train_size + val_size:])]

    return train, valid, test