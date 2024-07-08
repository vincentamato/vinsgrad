import unittest
import numpy as np
from vinsgrad.core.engine import Tensor, set_grad_enabled, no_grad

class TestTensor(unittest.TestCase):
    """
    Test cases for the Tensor class and related functions in engine.py.
    """

    def test_initialization(self):
        """
        Test the initialization of Tensor objects.
        """
        t = Tensor([1, 2, 3])
        self.assertEqual(t.shape, (3,))
        self.assertEqual(t.requires_grad, False)

        t = Tensor([[1, 2], [3, 4]], requires_grad=True)
        self.assertEqual(t.shape, (2, 2))
        self.assertEqual(t.requires_grad, True)

    def test_arithmetic_operations(self):
        """
        Test basic arithmetic operations on Tensor objects.
        """
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([4, 5, 6], requires_grad=True)

        # Addition
        c = a + b
        self.assertTrue(np.array_equal(c.data, [5, 7, 9]))
        self.assertTrue(c.requires_grad)

        # Subtraction
        d = a - b
        self.assertTrue(np.array_equal(d.data, [-3, -3, -3]))
        self.assertTrue(d.requires_grad)

        # Multiplication
        e = a * b
        self.assertTrue(np.array_equal(e.data, [4, 10, 18]))
        self.assertTrue(e.requires_grad)

        # Division
        f = a / b
        self.assertTrue(np.allclose(f.data, [0.25, 0.4, 0.5]))
        self.assertTrue(f.requires_grad)

    def test_matmul(self):
        """
        Test matrix multiplication of Tensor objects.
        """
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b = Tensor([[5, 6], [7, 8]], requires_grad=True)
        c = a @ b
        self.assertTrue(np.array_equal(c.data, [[19, 22], [43, 50]]))
        self.assertTrue(c.requires_grad)

    def test_pow(self):
        """
        Test power operation on Tensor objects.
        """
        a = Tensor([1, 2, 3], requires_grad=True)
        b = a ** 2
        self.assertTrue(np.array_equal(b.data, [1, 4, 9]))
        self.assertTrue(b.requires_grad)

    def test_reshape(self):
        """
        Test reshaping of Tensor objects.
        """
        a = Tensor([1, 2, 3, 4], requires_grad=True)
        b = a.reshape(2, 2)
        self.assertEqual(b.shape, (2, 2))
        self.assertTrue(np.array_equal(b.data, [[1, 2], [3, 4]]))
        self.assertTrue(b.requires_grad)

    def test_max(self):
        """
        Test max operation on Tensor objects.
        """
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b = a.max(axis=0)
        self.assertTrue(np.array_equal(b.data, [3, 4]))
        self.assertTrue(b.requires_grad)

    def test_exp(self):
        """
        Test exponential operation on Tensor objects.
        """
        a = Tensor([0, 1, 2], requires_grad=True)
        b = a.exp()
        self.assertTrue(np.allclose(b.data, [1, np.e, np.e**2]))
        self.assertTrue(b.requires_grad)

    def test_log(self):
        """
        Test natural logarithm operation on Tensor objects.
        """
        a = Tensor([1, np.e, np.e**2], requires_grad=True)
        b = a.log()
        self.assertTrue(np.allclose(b.data, [0, 1, 2]))
        self.assertTrue(b.requires_grad)

    def test_sum(self):
        """
        Test sum operation on Tensor objects.
        """
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b = a.sum(axis=0)
        self.assertTrue(np.array_equal(b.data, [4, 6]))
        self.assertTrue(b.requires_grad)

    def test_mean(self):
        """
        Test mean operation on Tensor objects.
        """
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b = a.mean(axis=1)
        self.assertTrue(np.array_equal(b.data, [1.5, 3.5]))
        self.assertTrue(b.requires_grad)

    def test_transpose(self):
        """
        Test transpose operation on Tensor objects.
        """
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b = a.T()
        self.assertTrue(np.array_equal(b.data, [[1, 3], [2, 4]]))
        self.assertTrue(b.requires_grad)

    def test_backward_add(self):
        """
        Test backward propagation for addition.
        """
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([4, 5, 6], requires_grad=True)
        c = a + b
        c.sum().backward()
        self.assertTrue(np.array_equal(a.grad, [1, 1, 1]))
        self.assertTrue(np.array_equal(b.grad, [1, 1, 1]))

    def test_backward_sub(self):
        """
        Test backward propagation for subtraction.
        """
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([4, 5, 6], requires_grad=True)
        c = a - b
        c.sum().backward()
        self.assertTrue(np.array_equal(a.grad, [1, 1, 1]))
        self.assertTrue(np.array_equal(b.grad, [-1, -1, -1]))

    def test_backward_mul(self):
        """
        Test backward propagation for multiplication.
        """
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([4, 5, 6], requires_grad=True)
        c = a * b
        c.sum().backward()
        self.assertTrue(np.array_equal(a.grad, [4, 5, 6]))
        self.assertTrue(np.array_equal(b.grad, [1, 2, 3]))

    def test_backward_div(self):
        """
        Test backward propagation for division.
        """
        a = Tensor([1, 4, 9], requires_grad=True)
        b = Tensor([1, 2, 3], requires_grad=True)
        c = a / b
        c.sum().backward()
        self.assertTrue(np.allclose(a.grad, [1, 0.5, 1/3]))
        self.assertTrue(np.allclose(b.grad, [-1, -1, -1]))

    def test_backward_matmul(self):
        """
        Test backward propagation for matrix multiplication.
        """
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b = Tensor([[5, 6], [7, 8]], requires_grad=True)
        c = a @ b
        c.sum().backward()
        self.assertTrue(np.array_equal(a.grad, [[11, 15], [11, 15]]))
        self.assertTrue(np.array_equal(b.grad, [[4, 4], [6, 6]]))

    def test_backward_pow(self):
        """
        Test backward propagation for power operation.
        """
        a = Tensor([1, 2, 3], requires_grad=True)
        b = a ** 2
        b.sum().backward()
        self.assertTrue(np.array_equal(a.grad, [2, 4, 6]))

    def test_backward_reshape(self):
        """
        Test backward propagation through reshape operation.
        """
        a = Tensor([1, 2, 3, 4], requires_grad=True)
        b = a.reshape(2, 2)
        b.sum().backward()
        self.assertTrue(np.array_equal(a.grad, [1, 1, 1, 1]))

    def test_backward_max(self):
        """
        Test backward propagation for max operation.
        """
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b = a.max(axis=0)
        b.sum().backward()
        self.assertTrue(np.array_equal(a.grad, [[0, 0], [1, 1]]))

    def test_backward_exp(self):
        """
        Test backward propagation for exponential operation.
        """
        a = Tensor([0, 1, 2], requires_grad=True)
        b = a.exp()
        b.sum().backward()
        self.assertTrue(np.allclose(a.grad, [1, np.e, np.e**2]))

    def test_backward_log(self):
        """
        Test backward propagation for natural logarithm operation.
        """
        a = Tensor([1, np.e, np.e**2], requires_grad=True)
        b = a.log()
        b.sum().backward()
        self.assertTrue(np.allclose(a.grad, [1, 1/np.e, 1/(np.e**2)]))

    def test_backward_sum(self):
        """
        Test backward propagation for sum operation.
        """
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b = a.sum(axis=0)
        b.sum().backward()
        self.assertTrue(np.array_equal(a.grad, [[1, 1], [1, 1]]))

    def test_backward_mean(self):
        """
        Test backward propagation for mean operation.
        """
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b = a.mean(axis=1)
        b.sum().backward()
        self.assertTrue(np.allclose(a.grad, [[0.5, 0.5], [0.5, 0.5]], atol=1e-6))

    def test_backward_transpose(self):
        """
        Test backward propagation for transpose operation.
        """
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b = a.T()
        b.sum().backward()
        self.assertTrue(np.array_equal(a.grad, [[1, 1], [1, 1]]))

    def test_set_grad_enabled(self):
        """
        Test the set_grad_enabled function.
        """
        prev_state = set_grad_enabled(False)
        a = Tensor([1, 2, 3], requires_grad=True)
        b = a * 2
        self.assertFalse(b.requires_grad)
        set_grad_enabled(prev_state)

    def test_no_grad(self):
        """
        Test the no_grad context manager.
        """
        a = Tensor([1, 2, 3], requires_grad=True)
        with no_grad():
            b = a * 2
        self.assertFalse(b.requires_grad)

    def test_broadcast(self):
        """
        Test broadcasting in arithmetic operations.
        """
        a = Tensor([[1, 2, 3]], requires_grad=True)
        b = Tensor([[1], [2], [3]], requires_grad=True)
        c = a + b
        self.assertEqual(c.shape, (3, 3))
        self.assertTrue(np.array_equal(c.data, [[2, 3, 4], [3, 4, 5], [4, 5, 6]]))
        self.assertTrue(c.requires_grad)

    def test_backward_broadcast(self):
        """
        Test backward propagation with broadcasting.
        """
        a = Tensor([[1, 2, 3]], requires_grad=True)
        b = Tensor([[1], [2], [3]], requires_grad=True)
        c = a + b
        c.sum().backward()
        self.assertTrue(np.array_equal(a.grad, [[3, 3, 3]]))
        self.assertTrue(np.array_equal(b.grad, [[3], [3], [3]]))

if __name__ == '__main__':
    unittest.main()