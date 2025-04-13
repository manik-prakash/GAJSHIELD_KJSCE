import torch.optim as optim
from torch.utils.data import DataLoader
from cnn_feature_extractor import EmberDualCNN
from ember_dataset import Ember2DImageDataset
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
batch_size = 32
learning_rate = 0.0005
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
train_dataset = Ember2DImageDataset("./data/ember2018/train_features_1.jsonl", max_samples=None)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)



# Initialize model, loss function, and optimizer
model = EmberDualCNN().to(device)
criterion = nn.BCEWithLogitsLoss()  # Binary classification loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for byteentropy, histogram, _, labels in train_loader:
        byteentropy, histogram, labels = (
            byteentropy.to(device),
            histogram.to(device),
            labels.float().unsqueeze(1).to(device),
        )

        # Forward pass
        outputs = model(byteentropy, histogram)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss and accuracy
        running_loss += loss.item()
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        total += labels.size(0)
        correct += (predictions == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

print("Training complete!")
torch.save(model.state_dict(), "EmberDualCNN.pth")  # Save the trained model