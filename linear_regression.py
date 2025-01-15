import numpy as np
import vinsgrad
import vinsgrad.nn as nn
import vinsgrad.optim as optim
from vinsgrad.utils.data import DataLoader

# Generate synthetic data
np.random.seed(0)
X = np.random.rand(1000, 1).astype(np.float32)
y = 2 * X + 1 + 0.1 * np.random.randn(1000, 1).astype(np.float32)

# Split data into train, validation, and test sets
train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))
test_size = len(X) - train_size - val_size

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

# Create data loaders
train_loader = DataLoader((X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader((X_val, y_val), batch_size=32, shuffle=False)
test_loader = DataLoader((X_test, y_test), batch_size=32, shuffle=False)

# Define the model
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

model = LinearRegression(input_dim=1, output_dim=1)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Helper function for evaluation
def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    
    for X_batch, y_batch in data_loader:
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss

# Training loop
num_epochs = 200
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (X_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Evaluate on validation set
    val_loss = evaluate(model, val_loader, criterion)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {running_loss/len(train_loader):.4f}, '
          f'Val Loss: {val_loss:.4f}')
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        vinsgrad.save(model.state_dict(), model_name='linear_regression', is_best=True)

# Final evaluation on test set
test_loss = evaluate(model, test_loader, criterion)
print(f'Test Loss: {test_loss:.4f}')

# Save final model
vinsgrad.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'test_loss': test_loss,
}, model_name='linear_regression')

print("Training completed!")