![logo](https://github.com/vincentamato/vinsgrad/blob/main/logo.png?raw=true)

A tiny deep learning library insipired by Karpathy's micrograd, smolorg's smolgrad, and Pytorch. I made this libary mainly as a means of learning about autograd, and I hope it will be as helpful for you as it has been for me. It covers the core mechanics of automatic differentiation and backpropagation which are the foundations for any modern deep learning library.

## Installation

```bash
pip install vinsgrad
```

## Automatic Differentiation and Backpropagation

Let's dive into the engine room of deep learning: automatic differentiation (autograd) and backpropagation. These are the unsung heroes that make training neural networks possible without losing our minds over partial derivatives.

### Automatic Differentiation

Imagine having a smart calculator that not only computes values but also keeps track of how to calculate derivatives for every single operation. That's essentially what automatic differentiation does. It breaks down complex functions into simple operations and applies the chain rule  of calculus as it goes along.

During the forward pass (when we're calculating the output of our model), autograd builds a computational graph. This graph is like a roadmap of all the operations performed, which comes in handy later when we need to calculate gradients.

### Backpropagation

Now, backpropagation is where the magic really happens. Once we have our computational graph from the forward pass, backprop starts at the end and works its way backwards (hence the name). It's like retracing your steps after a hike, but instead of looking for lost items, you're figuring out how much each step contributed to the final destination.

Backprop uses the chain rule to compute gradients for each operation in reverse order. It's efficient because it reuses computations, saving us from redundant calculations.

### Why This Matters

Without autograd and backprop, training neural networks would be a nightmare. These techniques allow us to automatically compute gradients for complex models with millions of parameters. It's what enables us to use gradient descent and its variants to optimize our models.

Understanding these concepts is crucial for anyone diving into deep learning. They're the foundational tools that make it possible to experiment with and train sophisticated neural networks without getting bogged down in manual derivative calculations. Creating this package has been extremely helpful for me, and I urge you to take a look at the code if you are confused about what automatic differentiation and backpropagation look like under the hood.

## Example Usage: Creating and Training a Multi-Layer Perceptron (MLP)

Let's walk through creating a simple MLP and see autograd and backpropagation in action. We'll create a model to classify handwritten digits from the MNIST dataset.

First, let's import the necessary modules and set up our data:

```python
import vinsgrad
import vinsgrad.nn as nn
import vinsgrad.optim as optim
from vinsgrad.utils.data import DataLoader
from vinsgrad.vision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Flatten()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

```

Now let's define our MLP:
```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)  # 784 input features (28x28 pixels)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)    # 10 output classes (digits 0-9)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = MLP()
```
This MLP has three fully connected (Linear) layers with ReLU activations between them. The `forward` method defines how data flows through the network. Now let's set up our loss function and optimizer:
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

Finally, let's train the model:
```python
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()                # Zero out previous gradients
        outputs = model(images)              # Compute logits
        loss = criterion(outputs, labels)    # Compute loss
        loss.backward()                      # Compute gradients
        optimizer.step()                     # Udpate weights
        running_loss += loss.item()
        
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
```
Here's what's happening in each iteration:
1. We perform a forward pass through the model, computing the loss.
2. We zero out the gradients from the previous iteration.
3. We call `loss.backward()`, which triggers the autograd engine to compute gradients for all tensors involved in the computation of the loss.
4. We call `optimizer.step()`, which updates the model's parameters based on the computed gradients.
This process repeats for each batch in each epoch. The autograd engine keeps track of the computational graph during the forward pass, and the backpropagation algorithm efficiently computes the gradients during the backward pass.
By leveraging autograd and backpropagation, we can focus on defining our model architecture and training loop, while the framework handles the complex task of computing gradients automatically. This makes it much easier to experiment with different model architectures and optimization techniques.
This example demonstrates how autograd and backpropagation are used in practice when training a neural network. It shows the creation of a simple MLP, setting up the data, loss function, and optimizer, and then goes through the training loop, highlighting where autograd and backpropagation come into play.

## ⚠️ **Work in Progress Alert!** 
Remember, vinsgrad is still in development and is a learning project. It might have quirks, bugs, or unexpected features (let's call them "surprise learning opportunities"). Use it to explore and learn, but maybe don't bet your PhD thesis on it just yet! Keep an eye out for updates and happy learning!
