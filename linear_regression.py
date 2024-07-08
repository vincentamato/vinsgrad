import numpy as np
import vinsgrad.nn as nn
import vinsgrad.optim as optim
from vinsgrad.core import Tensor

# Generate synthetic data
np.random.seed(0)
X_train = np.random.rand(100, 1).astype(np.float32)
y_train = 2 * X_train + 1 + 0.1 * np.random.randn(100, 1).astype(np.float32)

# Convert to vinsgrad tensors
X_train_tensor = Tensor(X_train)
y_train_tensor = Tensor(y_train)

# Define the model
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

model = LinearRegression(input_dim=1, output_dim=1)

# Define the custom MSELoss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Train the model
num_epochs = 200

for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    # Print the average loss for the epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')