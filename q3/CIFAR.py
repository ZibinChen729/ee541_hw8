#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

class CIFARMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3072, 256)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop2(x)

        x = self.fc3(x)
        return x

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

model = CIFARMLP()
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    weight_decay=0.0001
)

num_epochs = 40

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
ax.set_ylabel("Loss")
ax.set_title("CIFAR-10 MLP Loss")
ax.legend()
ax.grid(True)
plt.show()
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, train_accuracies, label="Train Accuracy")
ax.plot(epochs, test_accuracies, label="Test Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_title("CIFAR-10 MLP Accuracy")
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

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_xlabel("Predicted Class")
ax.set_ylabel("True Class")
ax.set_title("CIFAR-10 Confusion Matrix")
plt.show()
plt.close(fig)

most_confused_for_each_class = {}

for i in range(num_classes):
    row = cm[i].copy()
    row[i] = -1
    confused_class = np.argmax(row)
    most_confused_for_each_class[class_names[i]] = class_names[confused_class]

print("\nMost likely confused class for each object type:")
for true_class, confused_class in most_confused_for_each_class.items():
    print(f"{true_class} -> {confused_class}")

cm_no_diag = cm.copy()
np.fill_diagonal(cm_no_diag, 0)

max_index = np.unravel_index(np.argmax(cm_no_diag), cm_no_diag.shape)
true_class_idx, pred_class_idx = max_index

print("\nMost confused overall (one direction):")
print(f"True class: {class_names[true_class_idx]}")
print(f"Predicted as: {class_names[pred_class_idx]}")
print(f"Count: {cm_no_diag[true_class_idx, pred_class_idx]}")

max_pair = None
max_pair_value = -1

for i in range(num_classes):
    for j in range(i + 1, num_classes):
        pair_value = cm[i, j] + cm[j, i]
        if pair_value > max_pair_value:
            max_pair_value = pair_value
            max_pair = (i, j)

i, j = max_pair
print("\nMost mutually confused pair:")
print(f"{class_names[i]} and {class_names[j]}")
print(f"Total confusion: {max_pair_value}")


# Model description
# 
# An MLP was trained on the CIFAR-10 dataset. The input images were flattened into 3072-dimensional vectors. The model used two hidden layers with 256 and 128 nodes, respectively, with ReLU activation after each hidden layer. A dropout layer with rate 0.3 was added after each hidden layer output, and L2 regularization with coefficient 0.0001 was implemented through weight decay in SGD. The final output layer had 10 units corresponding to the 10 CIFAR-10 classes.
#     

# Training result analysis
# 
# The training loss and test loss both decreased during training, while the training and test accuracy both increased steadily. The final test accuracy was about 50.7%, which is reasonable for a fully connected MLP on CIFAR-10. Since CIFAR-10 is a more complex image dataset and the model does not use convolutional layers, its performance is naturally much lower than on MNIST. The training and test curves remained relatively close, which suggests that dropout and L2 regularization helped reduce overfitting.

# Confusion matrix analysis
# 
# The confusion matrix shows meaningful classification patterns. Several classes were classified relatively well, such as automobile, deer, frog, horse, ship, and truck. At the same time, there were several intuitive confusions between visually similar classes. For example, airplane was often confused with ship, automobile with truck, and cat with dog. These mistakes are expected because the model is a simple MLP and cannot capture local spatial features as effectively as a convolutional neural network.

# Part (a)
# 
# The most likely confused class for each object type is:
# 
# airplane → ship
# automobile → truck
# bird → deer
# cat → deer
# deer → frog
# dog → cat
# frog → deer
# horse → deer
# ship → truck
# truck → automobile

# Part (b)
# 
# The two classes most likely to be confused overall are cat and dog. This is reasonable because they share similar visual features, and a fully connected MLP has limited ability to distinguish fine local patterns in natural images.

# In[ ]:




