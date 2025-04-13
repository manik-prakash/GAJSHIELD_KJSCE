import os
import json
import pandas as pd

jsonl_path = "./data/ember2018/train_features_1.jsonl"          # This needs to be changed

def load_jsonl_to_dataframe(jsonl_path):
    samples = []

    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            if obj['label'] is not None:
                label = obj.get("label", -1)
                if label == -1:
                    continue  # skip unlabeled

                general_features = [
                    obj['general']['size'],  # File size
                    obj['general']['vsize'],  # Virtual size
                    obj['general']['imports'],  # Number of imports
                    obj['general']['exports'],  # Number of exports
                    obj['strings']['entropy'],  # File entropy
                ]
                
                # Append sample data
                samples.append({
                    'histogram': obj['histogram'],         # 256
                    'byteentropy': obj['byteentropy'],     # 256
                    'file_size': general_features[0],
                    'virtual_size': general_features[1],
                    'num_imports': general_features[2],
                    'num_exports': general_features[3],
                    'file_entropy': general_features[4],
                    'is_malware': 1 if label == 1 else 0,  # Binary label: 1 for malware, 0 for benign
                    'avClass': obj['avclass'] if label == 1 else "Benign"  # Malware family or "Benign"
                })

    # Convert the samples list to a pandas DataFrame
    df = pd.DataFrame(samples)

    return df


def save_dataframe_to_csv(df, csv_path):
    # Save the DataFrame to a CSV file
    df.to_csv(csv_path, index=False)

# Save the DataFrame to a CSV file
# output_csv_path = './data/train_features_1.csv'
# df.to_csv(output_csv_path, index=False)

csv_path = "output_data.csv"

# Load data into a DataFrame
df = load_jsonl_to_dataframe(jsonl_path)


print(df.head())   

# Save the DataFrame to a CSV file
save_dataframe_to_csv(df, csv_path)

print(f"CSV file saved to {csv_path}")