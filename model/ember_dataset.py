import json
import torch
from torch.utils.data import Dataset
import numpy as np

class Ember2DImageDataset(Dataset):
    def __init__(self, jsonl_path, max_samples=None): 
        self.samples = []
        self.labels = []

        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                obj = json.loads(line)
                if obj['label'] is not None:
                    label = obj.get("label", -1)
                    if label == -1:
                        continue  # skip unlabeled
                    # if label == 1:
                    #     print("Label 1 found")
                    self.samples.append({
                        'histogram': obj['histogram'],         # 256
                        'byteentropy': obj['byteentropy'],     # 256
                        'label': obj['label']
                    })
                    self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        hist = np.array(item['histogram'], dtype=np.float32).reshape(16, 16)
        byteentropy = np.array(item['byteentropy'], dtype=np.float32).reshape(16, 16)
        metadata_features = [
                item.get("size", 0),               # File size
                item.get("vsize", 0),             # Virtual size
                item.get("strings", {}).get("numstrings", 0),  # Number of strings
                item.get("imports", {}).get("num_imports", 0), # Number of imports
                item.get("exports", {}).get("num_exports", 0)  # Number of exports
        ]
        
        hist_tensor = torch.tensor(hist).unsqueeze(0)  # (1, 16, 16)
        byteentropy_tensor = torch.tensor(byteentropy).unsqueeze(0)  # (1, 16, 16)
        metadata_tensor = torch.tensor(metadata_features, dtype=torch.float32)



        label = torch.tensor(item['label'], dtype=torch.bool)

        return byteentropy_tensor, hist_tensor, label
