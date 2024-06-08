import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

import optuna

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Digit classification on MNIST dataset using feed forward neural network

# Device configuration
# Guarantees that the code will run on the GPU if it is available, otherwise it will run on the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fixed Hyperparameters
input_size = 784 # Image size: 28x28. We will flatten the image to a 1D-tensor of size 784
num_classes = 10 # Number of output classes (digits)
num_epochs = 2
batch_size = 100

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root="./MNIST_data", train=True,
                                            transform=torchvision.transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root="./MNIST_data", train=False,
                                            transform=torchvision.transforms.ToTensor())

# Data loader
# Use two seperate data loaders for train and test data
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False) # Shuffle doesn't matter for test data

# Split training data into training and validation data
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__() # Necessary when inheriting from nn.Module

        # Neural network architecture
        self.fc1 = nn.Linear(input_size, hidden_size) # First layer: input_size -> hidden_size
        self.relu = nn.ReLU() # Activation function
        self.fc2 = nn.Linear(hidden_size, num_classes) # Second layer: hidden_size -> num_classes

    # Forward pass
    def forward(self, x):
        # Pass input x through the layers
        # x is the input image, which is flattened to a 1D-tensor (size 784)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        # No softmax after the last layer (as actually necessary for multiclass classification),
        # because we are using CrossEntropyLoss (which applies softmax and log-likelihood loss itself)
        return out
    
def objective(trial):
    # Hyperparameters
    hidden_size = trial.suggest_int("hidden_size", 50, 200)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)

    model = NeuralNet(input_size, hidden_size, num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train on training data and validate on validation data

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    y_predicted = []
    y_true = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = model(images)

            _, predictions = torch.max(outputs, 1)

            y_predicted += predictions.cpu().tolist()
            y_true += labels.cpu().tolist()

    return accuracy_score(y_true, y_predicted)

# Run the optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

# Print the best hyperparameters
print(study.best_params)

# Train the model with the best hyperparameters

hidden_size, learning_rate = study.best_params["hidden_size"], study.best_params["learning_rate"]

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop

n_total_steps = len(train_loader) # Number of batches

for epoch in range(num_epochs):
    # Batches are of form (images, labels) with images of shape (batch_size, 1, 28, 28) and labels of shape (batch_size)
    for i, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)
        images = images.reshape(-1, 28*28).to(device) # -1 means that the first dimension will be inferred from the remaining dimensions
        labels = labels.to(device) # Important to move everything to the device

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad() # Set gradients to zero before backpropagation
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Step {i+1}/{n_total_steps}, Loss = {loss.item()}")

# Test the model
# In test phase, we don't need to compute gradients
y_predicted = []
y_true = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        # Returns the maximum value and the index of the maximum value, 1 is the dimension in which to take the max (the columns)
        _, predictions = torch.max(outputs, 1)

        # Append the predictions and the true labels to the lists
        y_predicted += predictions.cpu().tolist()
        y_true += labels.cpu().tolist()

# Show classification report and confusion matrix
print("Classification report:")
print(classification_report(y_true, y_predicted, labels=[i for i in range(10)]))
print("Confusion matrix:")
print(confusion_matrix(y_true, y_predicted))