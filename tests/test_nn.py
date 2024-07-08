import unittest
import numpy as np
import torch
import torch.nn as nn
from vinsgrad.nn import Module, Linear, ReLU, MSELoss, CrossEntropyLoss
from vinsgrad.core import Tensor

class TestModule(unittest.TestCase):
    print(np.__version__)
    def test_train_eval(self):
        """
        Test the train and eval modes of a simple neural network.
        """
        class TestNet(Module):
            def __init__(self):
                super().__init__()
                self.linear = Linear(10, 5)
                self.relu = ReLU()

        net = TestNet()
        torch_net = nn.Sequential(nn.Linear(10, 5), nn.ReLU())

        self.assertEqual(net.training, torch_net.training)
        
        net.eval()
        torch_net.eval()
        self.assertEqual(net.training, torch_net.training)

        net.train()
        torch_net.train()
        self.assertEqual(net.training, torch_net.training)

    def test_parameters(self):
        """
        Test if the parameters of a neural network are correctly identified and require gradients.
        """
        class TestNet(Module):
            def __init__(self):
                super().__init__()
                self.linear1 = Linear(10, 5)
                self.linear2 = Linear(5, 1)

        net = TestNet()
        torch_net = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 1))

        params = net.parameters()
        torch_params = list(torch_net.parameters())

        self.assertEqual(len(params), len(torch_params))
        for param, torch_param in zip(params, torch_params):
            self.assertEqual(param.requires_grad, torch_param.requires_grad)

class TestLinear(unittest.TestCase):
    def test_forward(self):
        """
        Test the forward pass of a linear layer.
        """
        linear = Linear(10, 5)
        torch_linear = nn.Linear(10, 5)
        
        linear.weight.data = torch_linear.weight.detach().numpy()
        linear.bias.data = torch_linear.bias.detach().numpy()

        x = Tensor(np.random.randn(3, 10))
        torch_x = torch.tensor(x.data, dtype=torch.float32)

        y = linear(x)
        torch_y = torch_linear(torch_x)

        np.testing.assert_allclose(y.data, torch_y.detach().numpy(), rtol=1e-5, atol=1e-6)

    def test_backward(self):
        """
        Test the backward pass of a linear layer.
        """
        linear = Linear(10, 5)
        torch_linear = nn.Linear(10, 5)
        
        linear.weight.data = torch_linear.weight.detach().numpy()
        linear.bias.data = torch_linear.bias.detach().numpy()

        x = Tensor(np.random.randn(3, 10), requires_grad=True)
        torch_x = torch.tensor(x.data, dtype=torch.float32, requires_grad=False)

        y = linear(x)
        torch_y = torch_linear(torch_x)

        z = y.sum()
        torch_z = torch_y.sum()

        z.backward()
        torch_z.backward()

        np.testing.assert_allclose(linear.weight.grad, torch_linear.weight.grad.detach().numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(linear.bias.grad, torch_linear.bias.grad.detach().numpy(), rtol=1e-5, atol=1e-6)

class TestReLU(unittest.TestCase):
    def test_forward(self):
        """
        Test the forward pass of a ReLU activation.
        """
        relu = ReLU()
        torch_relu = nn.ReLU()

        x = Tensor(np.array([-1, 0, 1]))
        torch_x = torch.tensor(x.data, dtype=torch.float32)

        y = relu(x)
        torch_y = torch_relu(torch_x)

        np.testing.assert_allclose(y.data, torch_y.numpy(), rtol=1e-5, atol=1e-6)
    def test_backward(self):
        """
        Test the backward pass of a ReLU activation.
        """
        relu = ReLU()
        torch_relu = nn.ReLU()

        x = Tensor(np.array([-1, 0, 1]), requires_grad=True)
        torch_x = torch.tensor(x.data, dtype=torch.float32, requires_grad=True)

        y = relu(x)
        torch_y = torch_relu(torch_x)

        z = y.sum()
        torch_z = torch_y.sum()

        z.backward()
        torch_z.backward()

        np.testing.assert_allclose(x.grad, torch_x.grad.numpy(), rtol=1e-5, atol=1e-6)

class TestMSELoss(unittest.TestCase):
    def test_forward(self):
        """
        Test the forward pass of Mean Squared Error loss.
        """
        mse = MSELoss()
        torch_mse = nn.MSELoss()

        y_pred = Tensor(np.array([1, 2, 3]))
        y_true = Tensor(np.array([2, 2, 2]))
        torch_y_pred = torch.tensor(y_pred.data, dtype=torch.float32)
        torch_y_true = torch.tensor(y_true.data, dtype=torch.float32)

        loss = mse(y_pred, y_true)
        torch_loss = torch_mse(torch_y_pred, torch_y_true)
        
        np.testing.assert_allclose(loss.data, torch_loss.detach().numpy(), rtol=1e-5, atol=1e-6)

    def test_backward(self):
        """
        Test the backward pass of Mean Squared Error loss.
        """
        mse = MSELoss()
        torch_mse = nn.MSELoss()

        y_pred = Tensor(np.array([1, 2, 3]), requires_grad=True)
        y_true = Tensor(np.array([2, 2, 2]))
        torch_y_pred = torch.tensor(y_pred.data, dtype=torch.float32, requires_grad=True)
        torch_y_true = torch.tensor(y_true.data, dtype=torch.float32)

        loss = mse(y_pred, y_true)
        torch_loss = torch_mse(torch_y_pred, torch_y_true)

        loss.backward()
        torch_loss.backward()

        np.testing.assert_allclose(y_pred.grad, torch_y_pred.grad.numpy(), rtol=1e-5, atol=1e-6)

class TestCrossEntropyLoss(unittest.TestCase):
    def test_forward(self):
        """
        Test the forward pass of Cross Entropy loss.
        """
        ce = CrossEntropyLoss()
        torch_ce = nn.CrossEntropyLoss()

        y_pred = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
        y_true = Tensor(np.array([[0, 1, 0], [1, 0, 0]]))
        torch_y_pred = torch.tensor(y_pred.data, dtype=torch.float32)
        torch_y_true = torch.tensor([1, 0]) 

        loss = ce(y_pred, y_true)
        torch_loss = torch_ce(torch_y_pred, torch_y_true)

        self.assertAlmostEqual(loss.item(), torch_loss.item(), places=6)

    def test_backward(self):
        """
        Test the backward pass of Cross Entropy loss.
        """
        ce = CrossEntropyLoss()
        torch_ce = nn.CrossEntropyLoss()

        y_pred = Tensor(np.array([[1, 2, 3], [4, 5, 6]]), requires_grad=True)
        y_true = Tensor(np.array([[0, 1, 0], [1, 0, 0]]))
        torch_y_pred = torch.tensor(y_pred.data, dtype=torch.float32, requires_grad=True)
        torch_y_true = torch.tensor([1, 0])  

        loss = ce(y_pred, y_true)
        torch_loss = torch_ce(torch_y_pred, torch_y_true)

        loss.backward()
        torch_loss.backward()

        np.testing.assert_allclose(loss.data, torch_loss.detach().numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(y_pred.grad, torch_y_pred.grad.numpy(), rtol=1e-5, atol=1e-6)

if __name__ == '__main__':
    unittest.main()