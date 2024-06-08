# Steps:
# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop:
# - forward pass: compute prediction
# - backward pass: gradients
# - update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0) prepare data
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
# X = np.linspace(-3, 12, 100)
# y = np.power(X, 2) + 4 * np.random.randn(100) - 7

# X = X.reshape(-1, 1)

# Convert to float tensor
X = torch.from_numpy(X.astype(np.float32))
y = torch.from_numpy(y.astype(np.float32))
y = y.view(y.shape[0], 1) # Shape: (n_samples, 1), view is similar to reshape

n_samples, n_features = X.shape

# 1) model
input_size = n_features
output_size = 1

model = nn.Linear(input_size, output_size)

# class NeuralNet(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(NeuralNet, self).__init__()
#         self.linear1 = nn.Linear(input_size, 6)
#         self.linear2 = nn.Linear(6, output_size)

#     def forward(self, x):
#         x = torch.relu(self.linear1(x))
#         x = self.linear2(x)
#         return x
    
# model = NeuralNet(input_size, output_size)

# 2) loss and optimizer
learning_rate = 0.01 # 0.001
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # Adam

# 3) training loop
#num_epochs = 3000
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass
    y_predicted = model(X)
    loss = criterion(y_predicted, y)

    # backward pass
    loss.backward()

    # update
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item()}')

# plot
# detach() to disable tracking in computation graph / autograd, generates new tensor with requires_grad=False
predicted = model(X).detach().numpy()
plt.plot(X, y, 'ro')
plt.plot(X, predicted, 'b')
plt.show()