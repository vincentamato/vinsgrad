# ![logo](https://github.com/vincentamato/vinsgrad/blob/main/logo.png?raw=true)

# vinsgrad: A Tiny Deep Learning Library

**vinsgrad** is a lightweight deep learning library inspired by [Karpathy's micrograd](https://github.com/karpathy/micrograd) and PyTorch. Created as a learning tool, it aims to simplify the understanding of **automatic differentiation** and **backpropagation**, the foundational techniques behind modern deep learning. Whether you're just starting out or looking to deepen your knowledge, **vinsgrad** provides a hands-on way to explore the mechanics of neural networks.

---

## Installation

Install vinsgrad via pip:
```bash
pip install vinsgrad
```

---

## Automatic Differentiation and Backpropagation

### Overview

The heart of any deep learning framework lies in its ability to compute gradients efficiently. **Automatic differentiation (autograd)** and **backpropagation** are the unsung heroes that make this possible. Here’s how these concepts work:

#### Automatic Differentiation

Think of autograd as a sophisticated calculator that not only evaluates expressions but also tracks the derivatives of every operation. This process:

- **Forward Pass**: Constructs a computational graph during the evaluation of a model's output.
- **Backward Pass**: Leverages the computational graph to calculate gradients using the chain rule efficiently.

This enables the training of models with millions of parameters without manually computing partial derivatives.

#### Backpropagation

Backpropagation works by retracing the computational graph built during the forward pass. It applies the chain rule in reverse to compute gradients for each operation. This efficient process avoids redundant calculations by reusing intermediate results, making it the backbone of gradient-based optimization techniques like stochastic gradient descent (SGD).

### Why It Matters

Autograd and backpropagation automate the gradient computation process, enabling rapid experimentation and optimization of neural network architectures. **vinsgrad** aims to make these concepts transparent and accessible, helping users understand the inner workings of deep learning.

---

## Example: Building and Training a Multi-Layer Perceptron (MLP)

This example demonstrates how to use **vinsgrad** to build and train a neural network for classifying handwritten digits from the MNIST dataset.

### Step 1: Set Up Data

```python
import vinsgrad
import vinsgrad.nn as nn
import vinsgrad.optim as optim
from vinsgrad.utils.data import DataLoader
from vinsgrad.vision import datasets, transforms

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Flatten()
])

# Load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

### Step 2: Define the Model

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)  # Input: 784 features (28x28 pixels)
        self.fc2 = nn.Linear(128, 64)  # Hidden layer: 128 -> 64
        self.fc3 = nn.Linear(64, 10)   # Output: 10 classes (digits 0-9)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the model
model = MLP()
```

### Step 3: Set Up Loss and Optimizer

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

### Step 4: Train the Model

```python
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        # Zero the gradients from the previous step
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
```

### Explanation
1. **Forward Pass**: Data flows through the model to compute predictions.
2. **Loss Computation**: The difference between predictions and ground truth is quantified.
3. **Gradient Calculation**: Using autograd, gradients are computed for each parameter.
4. **Parameter Update**: The optimizer updates model parameters based on gradients.

This process, repeated for each batch in each epoch, showcases the power of autograd and backpropagation.

---

## Work in Progress ✨

**vinsgrad** is an ongoing project designed primarily as a learning tool. While functional, it may have limitations or quirks. Your feedback and contributions are welcome as the library continues to evolve. Explore the code to gain insights into the fundamentals of automatic differentiation and backpropagation.

---

## Contributing

Contributions to **vinsgrad** are highly encouraged! Whether it’s fixing bugs, optimizing performance, or adding new features, your help is invaluable. Please refer to the [Contributing Guide](https://github.com/vincentamato/vinsgrad/blob/main/CONTRIBUTING.md) for more details.

---

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/vincentamato/vinsgrad/blob/main/LICENSE) file for details.

---

## Acknowledgements

Special thanks to [Andrej Karpathy](https://github.com/karpathy) for **micrograd** and the PyTorch team for inspiring this journey into deep learning fundamentals.

Happy learning and experimenting!