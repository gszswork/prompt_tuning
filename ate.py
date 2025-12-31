
import numpy as np
import pandas as pd
from causalnlp import CausalInferenceModel
import os, dotenv
import json
from pathlib import Path
import warnings

# Suppress LightGBM warnings
warnings.filterwarnings('ignore', message='.*Stopped training because there are no more leaves that meet the split requirements.*')
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')

dotenv.load_dotenv()

def binarize_column(data, column):
    mean_value = data[column].mean()
    return np.where(data[column] > mean_value, 1, 0)


def calculate_ate(df: pd.DataFrame, col_indices: list[int], ignore_cols: list[str]):
    """
    Calculate ATE for each feature column.
    
    Args:
        df: DataFrame with data
        col_indices: List of column indices to calculate ATE for
        ignore_cols: List of column names to ignore in the causal model
    
    Returns:
        List of dictionaries with 'feature' and 'ate' keys
    """
    results = []
    for col_index in col_indices:
        treatment_col = df.columns[col_index]

        df_copy = df.copy()
        df_copy[treatment_col] = binarize_column(df_copy, treatment_col)

        cm = CausalInferenceModel(
            df_copy,
            metalearner_type="t-learner",
            treatment_col=treatment_col,
            outcome_col="correct",
            ignore_cols=ignore_cols,
        ).fit()

        ate = cm.estimate_ate()
        results.append({
            'feature': treatment_col,
            'ate': ate
        })
    
    return results


if __name__ == "__main__":
    import glob
    
    # Find all CSV files in results/TextGrad
    csv_files = glob.glob("results/TextGrad/*.csv")
    
    if not csv_files:
        print("No CSV files found in results/TextGrad/")
        exit(1)
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    all_results = {}
    
    # Process each CSV file
    for csv_file in csv_files:
        print(f"Processing {csv_file}...")
        
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Check if required columns exist
            if 'prompt' not in df.columns or 'correct' not in df.columns:
                print(f"Warning: Required columns not found in {csv_file}, skipping...")
                continue
            
            # Find the index of 'prompt' column
            prompt_col_index = df.columns.get_loc('prompt')
            
            # Get all feature columns (those after 'prompt')
            feature_cols = df.columns[prompt_col_index + 1:].tolist()
            
            # Filter out non-numeric columns (keep only feature columns that are numeric)
            feature_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
            
            if not feature_cols:
                print(f"Warning: No numeric feature columns found in {csv_file}, skipping...")
                continue
            
            # Get column indices for feature columns
            feature_indices = [df.columns.get_loc(col) for col in feature_cols]
            
            # Define columns to ignore (non-feature columns)
            ignore_cols = [col for col in df.columns if col not in feature_cols and col != 'correct']
            
            # Calculate ATE for each feature
            results = calculate_ate(df, feature_indices, ignore_cols)
            
            # Convert results list to dictionary format (feature -> ate)
            ate_results = {}
            for result in results:
                ate_results[result['feature']] = result['ate']
            
            # Store results in structured format
            csv_path = Path(csv_file)
            all_results[csv_path.stem] = {
                "file": csv_path.name,
                "num_rows": len(df),
                "num_features": len(feature_cols),
                "ate_results": ate_results
            }
            
            print(f"Successfully processed {csv_file}: {len(results)} features analyzed")
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            import traceback
            traceback.print_exc()
            csv_path = Path(csv_file)
            all_results[csv_path.stem] = {
                "file": csv_path.name,
                "error": str(e),
                "status": "failed"
            }
    
    # Save results to JSON file
    output_file = Path('./Text_Grad.json')
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"All Results Summary")
    print(f"{'='*80}")
    print(f"Total files processed: {len(all_results)}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")
    
    # Calculate total features analyzed
    total_features = sum(
        result.get('num_features', 0) 
        for result in all_results.values() 
        if 'num_features' in result
    )
    print(f"Total features analyzed: {total_features}")
