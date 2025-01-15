import vinsgrad
import vinsgrad.nn as nn
import vinsgrad.optim as optim
from vinsgrad.vision import datasets, transforms
from vinsgrad.utils.data import DataLoader
from tqdm import tqdm

# Define the transforms
transform_pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the MNIST datasets
train_dataset = datasets.MNIST(root='./mnist_data', train=True, download=True, transform=transform_pipeline)
test_dataset = datasets.MNIST(root='./mnist_data', train=False, download=True, transform=transform_pipeline)

# Split the train dataset into train and validation sets
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = vinsgrad.utils.data.random_split(train_dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Flatten and fully connected layers
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

model = CNN()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Helper function for evaluation
def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with tqdm(data_loader, desc='Evaluating') as pbar:
        for images, labels in pbar:
            with vinsgrad.no_grad():
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                # Get predictions (outputs is batch_size x num_classes)
                # Get predicted class indices
                predictions = outputs.data.argmax(axis=1)
                # Get actual class indices from one-hot encoded labels
                actuals = labels.data.argmax(axis=1)
                
                # Update statistics
                total += labels.shape[0]
                correct += (predictions == actuals).sum()
                        
                pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    return avg_loss, accuracy

# Training loop
num_epochs = 10
best_val_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
        for images, labels in pbar:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'avg_loss': running_loss/(pbar.n + 1)
            })
    
    # Evaluate on validation set
    val_loss, val_accuracy = evaluate(model, val_loader, criterion)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {running_loss/len(train_loader):.4f}, '
          f'Val Loss: {val_loss:.4f}, '
          f'Val Accuracy: {val_accuracy:.4f}')
    
    # Save best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        vinsgrad.save(model.state_dict(), model_name='mnist_cnn', is_best=True)

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
}, model_name='mnist_cnn')

print("Training completed!")