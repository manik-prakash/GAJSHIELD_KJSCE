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

                    general_features = [
                        obj['general']['size'],  # File size
                        obj['general']['vsize'],  # Virtual size
                        obj['general']['imports'],  # Number of imports
                        obj['general']['exports'],  # Number of exports
                        obj['strings']['entropy'],  # File entropy
                        # obj['avclass']  # AVClass label (if available)
                    ]
                    # print(f"histogram: {type(histogram)}, byteentropy: {type(byteentropy)}")
                    self.samples.append({
                        'histogram': obj['histogram'],         # 256
                        'byteentropy': obj['byteentropy'],     # 256
                        'general_features': general_features,
                        'label': obj['label']
                    })
                    self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        hist = np.array(item['histogram'], dtype=np.float32).reshape(16, 16)
        byteentropy = np.array(item['byteentropy'], dtype=np.float32).reshape(16, 16)

        hist_tensor = torch.tensor(hist).unsqueeze(0)  # (1, 16, 16)
        byteentropy_tensor = torch.tensor(byteentropy).unsqueeze(0)  # (1, 16, 16)

        # Normalize tensors to 0–255
        hist_tensor = ((hist_tensor - hist_tensor.min()) / (hist_tensor.max() - hist_tensor.min())) * 255
        byteentropy_tensor = ((byteentropy_tensor - byteentropy_tensor.min()) / (byteentropy_tensor.max() - byteentropy_tensor.min())) * 255

        general_features = torch.tensor(item['general_features'], dtype=torch.float32)  # (N,)
        label = torch.tensor(item['label'], dtype=torch.bool)

        return byteentropy_tensor, hist_tensor, general_features, label
