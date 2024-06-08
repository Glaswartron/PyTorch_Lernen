import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")

# Hyperparameters
num_epochs = 10
batch_size = 8
learning_rate = 0.0015

# We are using the CIFAR-10 dataset

# dataset has PILImage images of range [0, 1]. We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load built-in CIFAR-10 dataset (torchvision)
train_dataset = torchvision.datasets.CIFAR10(root="./CIFAR10_data", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root="./CIFAR10_data", train=False, download=True, transform=transform)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Classes from CIFAR-10 dataset
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# Convolutional Neural Network class
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 3 input channels (RGB), 6 output channels, 5 kernel size
        # 6 output channels means that we have 6 filters / 6 feature maps
        # stride=1 -> kernel moves 1 pixel after each operation, padding=0 -> no zero-padding, padding>0 would add zeros around the image
        # We're doing a "3D" (2D with multiple channels) convolution,
        # because we have 3 input channels (there are different filters for each input channel)
        # This is done to detect different features in the image
        # The kernel size is the size of the filters that are applied to the image
        # This means we have 3 (channels) x 6 (filters) x 5x5 (kernel) + 6 (bias) = 456 parameters
        self.pool = nn.MaxPool2d(2, 2) # Max pooling with a 2x2 kernel
        # This reduces the size of the image by half
        # stride = 2 means that the kernel moves 2 pixels after each operation
        self.conv2 = nn.Conv2d(6, 16, 5) # 6 input channels (from previous layer), 16 output channels, 5 kernel size
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 16 feature maps, 5x5 image size after convolution and pooling, 120 output neurons
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # 10 output classes

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(-1, 16 * 5 * 5) # Flatten the image
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x


# Model
model = ConvNet().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss() # Also does softmax
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # Also try Adam optimizer

# Training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024 meaning 4 images, 3 channels (RGB), 32x32 pixels
        # input_layer: 3 input channels (RGB), 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item()}")

print("Finished training")

# Test the model
y_true = []
y_pred = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true += labels.tolist()
        y_pred += predicted.tolist()

print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
print(f"Classification Report: {classification_report(y_true, y_pred)}")
print(f"Confusion Matrix: {confusion_matrix(y_true, y_pred)}")
for i in range(10):
    print(f"Class {i} ({classes[i]}) accuracy: {accuracy_score(np.array(y_true)[np.array(y_true) == i], np.array(y_pred)[np.array(y_true) == i])}")

# Save the model checkpoint
torch.save(model.state_dict(), "model.ckpt")
# Saves the model's learned parameters
# Load like so: 
# model = ConvNet()
# model.load_state_dict(torch.load("model.ckpt"))