## Extract features for DSpy_miprov2 CSV files.
## This script processes full JSON prompt strings (including instructions, demos, etc.) to extract features.

import os
import pandas as pd
import dotenv
from llm_api import LLMClient
import json
import re

dotenv.load_dotenv()


class DataPipeline:
    def __init__(
        self,
        instructions: list[str],
        api_key: str,
        base_url: str,
        model: str = "gpt-5.2",
    ):
        self.instructions = instructions
        self.client = LLMClient(
            api_key=api_key,
            base_url=base_url,
            default_model=model
        )
        self.model = model

    def label_instructions(
        self,
        sample_instructs,
        features,
        instruction,
    ):
        messages = [
            {
                "role": "assistant",
                "content": str(sample_instructs),
            },
            {
                "role": "user",
                "content": f"What are the measurable and improveable textual features of the instructions generated above {sample_instructs}, for solving the ask of solving problems? Make sure these features are independent of each other and not confounded.  Give the answer directly without preparatory statements." ,
            },
            {
                "role": "assistant",
                "content": str(features),
            },
            {
                "role": "user",
                "content": f"According to the order of the factors:{features}, score the instruction:{instruction} with 1 to 10. The final result must be a string of scores separated by commas. Give the answer directly without preparatory statements.",
            },
        ]
        instruction_label = self.client.chat_with_messages(messages, model=self.model)
        return instruction_label

    def load_manual_features(self, json_path: str = "dynamic_feats.json") -> tuple[str, str, list[str]]:
        """
        Load features from the manual features JSON file.
        Returns:
            - features: Full feature descriptions string (for label_instructions)
            - show_features: Comma-separated feature names (for column names)
            - features_list: List of full feature descriptions (for label_instructions)
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        features_text = data["prompt"]
        
        # Parse feature names and descriptions
        # Features are formatted as "1. **FeatureName**: Description..."
        feature_pattern = r'\d+\.\s+\*\*([^*]+)\*\*:\s*(.+?)(?=\n\n\d+\.|\Z)'
        matches = re.findall(feature_pattern, features_text, re.DOTALL)
        
        feature_names = [match[0].strip() for match in matches]
        feature_descriptions = [f"{i+1}. **{match[0].strip()}**: {match[1].strip()}" 
                               for i, match in enumerate(matches)]
        
        # Create comma-separated feature names for show_features
        show_features = ", ".join(feature_names)
        
        # Create features_list (full descriptions) for label_instructions
        features_list = feature_descriptions
        
        # Create full features string (similar to what extract_features would return)
        features = "\n\n".join(feature_descriptions)
        
        return features, show_features, features_list

    def run(self) -> dict:
        """
        Run the pipeline to extract features for instructions.
        Returns a dictionary mapping prompt -> feature scores (as a dict).
        """
        instructions = self.instructions
        
        # Load manual features instead of using LLM
        features, show_features, features_list = self.load_manual_features()

        # Create a mapping of prompt -> feature scores
        prompt_features = {}
        
        for instruction in instructions:
            instruction_label = self.label_instructions(instructions, features_list, instruction)
            # Parse the scores
            scores = [s.strip() for s in instruction_label.split(",")]
            feature_names = [name.strip() for name in show_features.split(",")]
            
            # Create a dict mapping feature name -> score
            feature_dict = {}
            for i, feature_name in enumerate(feature_names):
                if i < len(scores):
                    feature_dict[feature_name] = scores[i]
            
            prompt_features[instruction] = feature_dict
        
        return prompt_features, show_features


def process_csv_file(csv_path: str, pipeline: DataPipeline):
    """
    Process a single DSpy CSV file: use full JSON prompt strings as unique identifiers, get features, and add feature columns.
    """
    print(f"Processing {csv_path}...")
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Check if 'prompt' column exists
    if 'prompt' not in df.columns:
        print(f"Warning: 'prompt' column not found in {csv_path}, skipping...")
        return
    
    # Get unique prompts (using the full JSON string as the unique identifier)
    unique_prompts = df['prompt'].unique().tolist()
    print(f"Found {len(unique_prompts)} unique prompts in {csv_path}")
    
    if len(unique_prompts) == 0:
        print(f"Warning: No prompts found in {csv_path}, skipping...")
        return
    
    # Get features for all unique prompts (using the full prompt string)
    pipeline.instructions = unique_prompts
    prompt_features, show_features = pipeline.run()
    
    # Extract feature names (original with spaces for matching, cleaned for column names)
    feature_names_original = [name.strip() for name in show_features.split(",") if name.strip()]
    feature_names_cleaned = [name.replace(" ", "") for name in feature_names_original]
    
    # Filter out empty feature names
    valid_indices = [i for i, name in enumerate(feature_names_cleaned) if name]
    feature_names_original = [feature_names_original[i] for i in valid_indices]
    feature_names_cleaned = [feature_names_cleaned[i] for i in valid_indices]
    
    # Create mapping from original to cleaned names
    name_mapping = dict(zip(feature_names_original, feature_names_cleaned))
    
    # Find the index of 'prompt' column
    prompt_idx = df.columns.get_loc('prompt')
    
    # Keep columns up to and including 'prompt', remove everything after it
    columns_to_keep = df.columns[:prompt_idx + 1].tolist()
    df = df[columns_to_keep].copy()
    
    # Create feature columns initialized with None (using cleaned names)
    for feature_name_cleaned in feature_names_cleaned:
        df.loc[:, feature_name_cleaned] = None
    
    # Map prompts to their features
    prompt_to_features = {}
    for prompt, features in prompt_features.items():
        feature_dict_cleaned = {}
        for feature_name_original, feature_name_cleaned in name_mapping.items():
            if feature_name_original in features:
                feature_dict_cleaned[feature_name_cleaned] = features[feature_name_original]
        prompt_to_features[prompt] = feature_dict_cleaned
    
    # Assign features using loc to avoid chained assignment warnings
    for idx in df.index:
        prompt = df.loc[idx, 'prompt']
        if prompt in prompt_to_features:
            features = prompt_to_features[prompt]
            for feature_name_cleaned, value in features.items():
                df.loc[idx, feature_name_cleaned] = value
    
    # Write back to the same CSV file
    df.to_csv(csv_path, index=False, sep=",")
    print(f"Successfully updated {csv_path} with feature columns")


if __name__ == "__main__":
    import glob
    
    # Find all CSV files in results/DSpy_miprov2
    csv_files = glob.glob("results/DSpy_miprov2/*.csv")
    
    if not csv_files:
        print("No CSV files found in results/DSpy_miprov2/")
        exit(1)
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Initialize pipeline (instructions will be set per file)
    pipeline = DataPipeline(
        instructions=[],  # Will be set per file
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.openai.com/v1",
        model="gpt-5.2",
    )
    
    # Process each CSV file
    for csv_file in csv_files:
        try:
            process_csv_file(csv_file, pipeline)
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            import traceback
            traceback.print_exc()
    
    print("All CSV files processed!")

