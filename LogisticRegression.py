# Steps:
# 0) Prepare data
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 0) prepare data
X, y = datasets.make_classification(n_samples=100, n_features=5, n_informative=3, n_redundant=2, n_clusters_per_class=1,
                                    shuffle=True, random_state=42)

n_samples, n_features = X.shape

# Train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Convert to tensors
# Should be done because PyTorch expects tensors
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# Reshape y
# We need to reshape y because PyTorch expects the shape (n_samples, 1) for y and current shape is (n_samples,) which is just a tuple
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# 1) model
# n_features > 1
input_size = n_features

class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        # Prediction is a probability (kind of), so we use sigmoid to squash it between 0 and 1
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted
    
model = LogisticRegression(input_size)

# 2) loss and optimizer
# BCE = Binary Cross Entropy is used for binary classification
# For multiclass classification, use CrossEntropyLoss
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training loop
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)

    # backward pass
    loss.backward()

    # update
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item()}')

# Evaluate predictions
with torch.no_grad():
    y_predicted = model(X_test)
    # y_predicted are probabilities, so we need to round them to get the class
    y_predicted_cls = y_predicted.round()
    # Accuracy
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy: {acc.item()}')