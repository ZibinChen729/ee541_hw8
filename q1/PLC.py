#!/usr/bin/env python
# coding: utf-8

# In[17]:


import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
class MNISTDataset(Dataset):
    def __init__(self, file_path):
        with h5py.File(file_path, "r") as f:
            self.x = f["xdata"][:]
            self.y = f["ydata"][:]


        self.x = self.x.astype(np.float32)


        if self.x.max() > 1:
            self.x = self.x / 255.0


        if len(self.y.shape) > 1:
            self.y = np.argmax(self.y, axis=1)

        self.y = self.y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

train_dataset = MNISTDataset("mnist_traindata.hdf5")
test_dataset = MNISTDataset("mnist_testdata.hdf5")

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

print("train x shape:", train_dataset.x.shape)
print("train y shape:", train_dataset.y.shape)
print("first y sample:", train_dataset.y[0])

x_batch, y_batch = next(iter(train_loader))
print("x_batch shape:", x_batch.shape)
print("y_batch shape:", y_batch.shape)
print("first 10 labels:", y_batch[:10])
print("y_batch dtype:", y_batch.dtype)

model = nn.Sequential(
    nn.Linear(28 * 28, 10)
)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    weight_decay=1e-4

)

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        optimizer.zero_grad()

        outputs = model(x)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            outputs = model(x)
            loss = criterion(outputs, y)

            total_loss += loss.item() * x.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy, np.array(all_labels), np.array(all_preds)


    num_epochs = 20

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion)

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
    print(f"Test  Loss: {test_loss:.4f}, Test  Accuracy: {test_acc:.4f}")
    print("-" * 40)

    epochs = range(1, num_epochs + 1)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, train_losses, label="Train Loss")
ax.plot(epochs, test_losses, label="Test Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Log-Loss")
ax.set_title("Training and Test Log-Loss")
ax.legend()
ax.grid(True)
plt.show()
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, train_accuracies, label="Train Accuracy")
ax.plot(epochs, test_accuracies, label="Test Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_title("Training and Test Accuracy")
ax.legend()
ax.grid(True)
plt.show()
plt.close(fig)

test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion)
print("Final Test Accuracy:", test_acc)

num_classes = 10
cm = np.zeros((num_classes, num_classes), dtype=np.int32)

for true_label, pred_label in zip(y_true, y_pred):
    cm[true_label, pred_label] += 1

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted Class")
ax.set_ylabel("True Class")
ax.set_title("Confusion Matrix")
plt.show()
plt.close(fig)


# Model description
# 
# A multiclass logistic classifier was implemented in PyTorch for the MNIST dataset. The model was built using nn.Sequential with a single fully connected linear layer that maps the 784-dimensional input vector to 10 output classes. No softmax activation was added at the output layer because CrossEntropyLoss already includes the softmax operation internally. The model was trained using stochastic gradient descent (SGD) with a mini-batch size of 100. L2 regularization was applied through weight decay to improve generalization.

# Training result analysis
# 
# The training and test log-loss both decreased steadily over the epochs, while the training and test accuracy both increased and then gradually stabilized. This indicates that the model learned the classification task effectively. The final test accuracy was about 92.4%, which is reasonable for a single-layer logistic classifier on MNIST. The training and test curves stayed close to each other, suggesting that the model did not suffer from severe overfitting.
# 

# Confusion matrix analysis
# 
# The confusion matrix is reasonable because most values are concentrated along the diagonal, which means the majority of test samples were correctly classified. Some confusion appears between visually similar digits, such as 3 and 5, 4 and 9, and 2 and 8. This is expected because some handwritten digits have similar shapes. Overall, the confusion matrix shows that the classifier learned meaningful decision boundaries for most classes.

# In[ ]:




