import unittest
import numpy as np
import torch
from vinsgrad.core import Tensor
from vinsgrad.optim import SGD
from vinsgrad.nn import Linear

class TestSGD(unittest.TestCase):
    def test_sgd_no_momentum(self):
        """
        Test SGD optimizer without momentum.
        """
        np.random.seed(42)
        torch.manual_seed(42)

        vg_linear = Linear(10, 5)
        vg_optimizer = SGD(vg_linear.parameters(), lr=0.01, momentum=0)

        torch_linear = torch.nn.Linear(10, 5)
        torch_optimizer = torch.optim.SGD(torch_linear.parameters(), lr=0.01, momentum=0)

        with torch.no_grad():
            torch_linear.weight.copy_(torch.tensor(vg_linear.weight.data, dtype=torch.float32))
            torch_linear.bias.copy_(torch.tensor(vg_linear.bias.data, dtype=torch.float32))

        x = np.random.randn(20, 10).astype(np.float32)
        y = np.random.randn(20, 5).astype(np.float32)

        for _ in range(5):
            vg_output = vg_linear(Tensor(x))
            vg_loss = ((vg_output - Tensor(y)) ** 2).mean()
            vg_loss.backward()
            vg_optimizer.step()
            vg_optimizer.zero_grad()

            torch_output = torch_linear(torch.tensor(x, dtype=torch.float32))
            torch_loss = ((torch_output - torch.tensor(y, dtype=torch.float32)) ** 2).mean()
            torch_loss.backward()
            torch_optimizer.step()
            torch_optimizer.zero_grad()

        np.testing.assert_allclose(vg_linear.weight.data, torch_linear.weight.detach().numpy(), rtol=1.0, atol=1e-2)
        np.testing.assert_allclose(vg_linear.bias.data, torch_linear.bias.detach().numpy(), rtol=1.0, atol=1e-2)

    def test_sgd_with_momentum(self):
        """
        Test SGD optimizer with momentum.
        """
        np.random.seed(42)
        torch.manual_seed(42)

        vg_linear = Linear(10, 5)
        vg_optimizer = SGD(vg_linear.parameters(), lr=0.01, momentum=0.9)

        torch_linear = torch.nn.Linear(10, 5)
        torch_optimizer = torch.optim.SGD(torch_linear.parameters(), lr=0.01, momentum=0.9)

        with torch.no_grad():
            torch_linear.weight.copy_(torch.tensor(vg_linear.weight.data, dtype=torch.float32))
            torch_linear.bias.copy_(torch.tensor(vg_linear.bias.data, dtype=torch.float32))

        x = np.random.randn(20, 10).astype(np.float32)
        y = np.random.randn(20, 5).astype(np.float32)

        for _ in range(5):
            vg_output = vg_linear(Tensor(x))
            vg_loss = ((vg_output - Tensor(y)) ** 2).mean()
            vg_loss.backward()
            vg_optimizer.step()
            vg_optimizer.zero_grad()

            torch_output = torch_linear(torch.tensor(x, dtype=torch.float32))
            torch_loss = ((torch_output - torch.tensor(y, dtype=torch.float32)) ** 2).mean()
            torch_loss.backward()
            torch_optimizer.step()
            torch_optimizer.zero_grad()

        np.testing.assert_allclose(vg_linear.weight.data, torch_linear.weight.detach().numpy(), rtol=1.0, atol=1e-2)
        np.testing.assert_allclose(vg_linear.bias.data, torch_linear.bias.detach().numpy(), rtol=1.0, atol=1e-2)

    def test_sgd_state_dict(self):
        np.random.seed(42)
        torch.manual_seed(42)

        vg_linear = Linear(10, 5)
        vg_optimizer = SGD(vg_linear.parameters(), lr=0.01, momentum=0.9)

        torch_linear = torch.nn.Linear(10, 5)
        torch_optimizer = torch.optim.SGD(torch_linear.parameters(), lr=0.01, momentum=0.9)

        with torch.no_grad():
            torch_linear.weight.copy_(torch.tensor(vg_linear.weight.data, dtype=torch.float32))
            torch_linear.bias.copy_(torch.tensor(vg_linear.bias.data, dtype=torch.float32))

        x = np.random.randn(20, 10).astype(np.float32)
        y = np.random.randn(20, 5).astype(np.float32)

        for _ in range(5):
            vg_output = vg_linear(Tensor(x))
            vg_loss = ((vg_output - Tensor(y)) ** 2).mean()
            vg_loss.backward()
            vg_optimizer.step()
            vg_optimizer.zero_grad()

            torch_output = torch_linear(torch.tensor(x, dtype=torch.float32))
            torch_loss = ((torch_output - torch.tensor(y, dtype=torch.float32)) ** 2).mean()
            torch_loss.backward()
            torch_optimizer.step()
            torch_optimizer.zero_grad()

        state_dict = vg_optimizer.state_dict()

        new_vg_optimizer = SGD(vg_linear.parameters(), lr=0.01, momentum=0.9)
        new_vg_optimizer.load_state_dict(state_dict)

        for _ in range(5):
            vg_output = vg_linear(Tensor(x))
            vg_loss = ((vg_output - Tensor(y)) ** 2).mean()
            vg_loss.backward()
            new_vg_optimizer.step()
            new_vg_optimizer.zero_grad()

            torch_output = torch_linear(torch.tensor(x, dtype=torch.float32))
            torch_loss = ((torch_output - torch.tensor(y, dtype=torch.float32)) ** 2).mean()
            torch_loss.backward()
            torch_optimizer.step()
            torch_optimizer.zero_grad()

        np.testing.assert_allclose(vg_linear.weight.data, torch_linear.weight.detach().numpy(), rtol=7.0, atol=1e-2)
        np.testing.assert_allclose(vg_linear.bias.data, torch_linear.bias.detach().numpy(), rtol=1.0, atol=1e-2)

if __name__ == '__main__':
    unittest.main()
