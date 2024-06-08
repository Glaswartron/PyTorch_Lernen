import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix

# Digit classification on MNIST dataset using feed forward neural network

# Device configuration
# Guarantees that the code will run on the GPU if it is available, otherwise it will run on the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 784 # Image size: 28x28. We will flatten the image to a 1D-tensor of size 784
hidden_size = 100 # Number of neurons in the hidden layer
num_classes = 10 # Number of output classes (digits)
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root="./MNIST_data", train=True,
                                            transform=torchvision.transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root="./MNIST_data", train=False,
                                            transform=torchvision.transforms.ToTensor())

# Data loader
# Use two seperate data loaders for train and test data
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False) # Shuffle doesn't matter for test data

# examples = iter(train_loader)
# samples, labels = next(examples)
# print(samples.shape, labels.shape)
# Gives: torch.Size([100, 1, 28, 28]) torch.Size([100])
# 100 samples per batch (batch_size), 1 channel (grayscale, no color channels), then actual image: 28x28 pixels

# See MNIST_Example.png
# for i in range(6):
#     plt.subplot(2, 3, i+1)
#     plt.imshow(samples[i][0], cmap="gray")
# plt.show()

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
    
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
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