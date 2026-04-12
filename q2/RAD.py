#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

class Model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 48)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(48, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
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

    with torch.no_grad():
        for x, y in loader:
            outputs = model(x)
            loss = criterion(outputs, y)

            total_loss += loss.item() * x.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

num_epochs = 40
criterion = nn.CrossEntropyLoss()

model1 = Model1()
optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.1, weight_decay=0.0)

model1_train_losses = []
model1_test_losses = []
model1_train_accs = []
model1_test_accs = []

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model1, train_loader, criterion, optimizer1)
    test_loss, test_acc = evaluate(model1, test_loader, criterion)

    model1_train_losses.append(train_loss)
    model1_test_losses.append(test_loss)
    model1_train_accs.append(train_acc)
    model1_test_accs.append(test_acc)

    print(f"Model 1 - Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Test  Loss: {test_loss:.4f}, Test  Acc: {test_acc:.4f}")
    print("-" * 40)

model2 = Model2()
optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.1, weight_decay=0.0001)

model2_train_losses = []
model2_test_losses = []
model2_train_accs = []
model2_test_accs = []

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model2, train_loader, criterion, optimizer2)
    test_loss, test_acc = evaluate(model2, test_loader, criterion)

    model2_train_losses.append(train_loss)
    model2_test_losses.append(test_loss)
    model2_train_accs.append(train_acc)
    model2_test_accs.append(test_acc)

    print(f"Model 2 - Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Test  Loss: {test_loss:.4f}, Test  Acc: {test_acc:.4f}")
    print("-" * 40)

epochs = range(1, num_epochs + 1)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, model1_train_losses, label="Train Loss")
ax.plot(epochs, model1_test_losses, label="Test Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Model 1 Loss")
ax.legend()
ax.grid(True)
plt.show()
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, model1_train_accs, label="Train Accuracy")
ax.plot(epochs, model1_test_accs, label="Test Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_title("Model 1 Accuracy")
ax.legend()
ax.grid(True)
plt.show()
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, model2_train_losses, label="Train Loss")
ax.plot(epochs, model2_test_losses, label="Test Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Model 2 Loss")
ax.legend()
ax.grid(True)
plt.show()
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, model2_train_accs, label="Train Accuracy")
ax.plot(epochs, model2_test_accs, label="Test Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_title("Model 2 Accuracy")
ax.legend()
ax.grid(True)
plt.show()
plt.close(fig)

w1_fc1 = model1.fc1.weight.detach().cpu().numpy().flatten()
w1_fc2 = model1.fc2.weight.detach().cpu().numpy().flatten()

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(w1_fc1, bins=50)
ax.set_title("Model 1 - Input Layer Weights")
ax.set_xlabel("Weight Value")
ax.set_ylabel("Frequency")
plt.show()
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(w1_fc2, bins=50)
ax.set_title("Model 1 - Hidden/Output Layer Weights")
ax.set_xlabel("Weight Value")
ax.set_ylabel("Frequency")
plt.show()
plt.close(fig)

w2_fc1 = model2.fc1.weight.detach().cpu().numpy().flatten()
w2_fc2 = model2.fc2.weight.detach().cpu().numpy().flatten()

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(w2_fc1, bins=50)
ax.set_title("Model 2 - Input Layer Weights")
ax.set_xlabel("Weight Value")
ax.set_ylabel("Frequency")
plt.show()
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(w2_fc2, bins=50)
ax.set_title("Model 2 - Hidden/Output Layer Weights")
ax.set_xlabel("Weight Value")
ax.set_ylabel("Frequency")
plt.show()
plt.close(fig)


# Model comparison.
# 
# Two neural networks were trained on Fashion MNIST for 40 epochs. Model 1 used one hidden layer with 128 ReLU units and did not use regularization or dropout. Model 2 used one hidden layer with 48 ReLU units, L2 regularization with coefficient 0.0001, and dropout with rate 0.2. Both models were trained using SGD and cross-entropy loss.

# Training result analysis
# 
# Model 1 achieved higher training accuracy and lower training loss than Model 2. However, the gap between training and test performance was also larger, which suggests that Model 1 had a stronger tendency to overfit. In contrast, Model 2 had slightly lower training performance, but the training and test curves stayed closer together. This indicates that regularization and dropout helped control overfitting and improved generalization stability.

# Histogram comparison
# 
# The weight histograms of Model 1 are more spread out, with a wider range of positive and negative values. In particular, the hidden/output layer of Model 1 contains more large-magnitude weights. In contrast, the weights of Model 2 are more concentrated around zero, especially in the input layer histogram. The distribution is narrower and contains fewer extreme values.

# Effect of regularization
# 
# Regularization makes the weight distribution more concentrated around zero. It reduces the magnitude of the weights and discourages excessively large positive or negative values. As a result, the model becomes less complex and is less likely to overfit the training data. Therefore, regularization helps improve generalization by shrinking the weights toward zero.

# In[ ]:




