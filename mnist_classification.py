import vinsgrad
import vinsgrad.nn as nn
import vinsgrad.optim as optim
from vinsgrad.vision import datasets, transforms
from vinsgrad.utils.data import DataLoader

# Define the transforms
transform_pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Flatten()
])


# Load the MNIST dataset
mnist_data = datasets.MNIST(transform=transform_pipeline)

# Split the dataset into train and test sets
train_images, train_labels = mnist_data.get_train_data()
test_images, test_labels = mnist_data.get_test_data()

# Split the train set into train and validation sets
train_size = int(0.8 * len(train_images))
val_size = len(train_images) - train_size
train_images, val_images = train_images[:train_size], train_images[train_size:]
train_labels, val_labels = train_labels[:train_size], train_labels[train_size:]

# Create data loaders
train_loader = DataLoader(train_images, train_labels, batch_size=64, shuffle=True)
val_loader = DataLoader(val_images, val_labels, batch_size=64, shuffle=False)
test_loader = DataLoader(test_images, test_labels, batch_size=64, shuffle=False)


# Define the model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

model = MLP()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Helper function for evaluation
def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in data_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        
        predicted = vinsgrad.argmax(outputs.data, axis=1)
        total += labels.shape[0]
        correct += (predicted == vinsgrad.argmax(labels.data, axis=1)).sum().item()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    return avg_loss, accuracy

# Training loop
num_epochs = 10
best_val_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Evaluate on validation set
    val_loss, val_accuracy = evaluate(model, val_loader, criterion)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {running_loss/len(train_loader):.4f}, '
          f'Val Loss: {val_loss:.4f}, '
          f'Val Accuracy: {val_accuracy:.4f}')
    
    # Save best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        vinsgrad.save(model.state_dict(), model_name='mnist_mlp', is_best=True)

# Final evaluation on test set
test_loss, test_accuracy = evaluate(model, test_loader, criterion)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

# Save final model
vinsgrad.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'test_loss': test_loss,
    'test_accuracy': test_accuracy
}, model_name='mnist_mlp')

print("Training completed!")