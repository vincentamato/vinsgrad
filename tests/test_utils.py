import unittest
import os
import shutil
import numpy as np
import torch
import vinsgrad
from vinsgrad import Tensor

class TestSaveLoadFunctions(unittest.TestCase):

    @classmethod
    def setUp(self):
        """
        Set up the test environment.
        """
        self.test_dir = 'test_checkpoints'
        os.makedirs(self.test_dir, exist_ok=True)

        self.test_obj = {'test': 'data'}
        self.model_name = 'test_model'
        self.epoch = 1

    @classmethod
    def tearDown(self):
        """
        Tear down the test environment.
        """
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_save_and_load(self):
        """
        Test saving and loading an object.
        """
        filename = vinsgrad.save(self.test_obj, self.model_name, epoch=self.epoch, dir_path=self.test_dir)
        loaded_obj = vinsgrad.load(filename)
        self.assertEqual(self.test_obj, loaded_obj)

    def test_save_best_model(self):
        """
        Test saving the best model.
        """
        best_filename = vinsgrad.save(self.test_obj, self.model_name, is_best=True, dir_path=self.test_dir)
        loaded_obj = vinsgrad.load(best_filename)
        self.assertEqual(self.test_obj, loaded_obj)

    def test_save_without_epoch(self):
        """
        Test saving without epoch.
        """
        filename = vinsgrad.save(self.test_obj, self.model_name, dir_path=self.test_dir)
        loaded_obj = vinsgrad.load(filename)
        self.assertEqual(self.test_obj, loaded_obj)

    def test_load_non_existent_file(self):
        """
        Test loading a non-existent file.
        """
        non_existent_file = 'non_existent_file.pkl.gz'
        with self.assertRaises(FileNotFoundError):
            vinsgrad.load(non_existent_file)

class TestArgmaxFunction(unittest.TestCase):
    
    @classmethod
    def setUp(self):
        """
        Set up the test environment.
        """
        self.data = np.random.rand(10, 20, 30)
        self.tensor = Tensor(self.data)
        self.torch_tensor = torch.tensor(self.data)

    def test_argmax_axis0(self):
        """
        Test argmax along axis 0.
        """
        vinsgrad_result = vinsgrad.argmax(self.tensor, axis=0).data
        torch_result = torch.argmax(self.torch_tensor, dim=0).numpy()
        np.testing.assert_array_equal(vinsgrad_result, torch_result)
    
    def test_argmax_axis1(self):
        """
        Test argmax along axis 1.
        """
        vinsgrad_result = vinsgrad.argmax(self.tensor, axis=1).data
        torch_result = torch.argmax(self.torch_tensor, dim=1).numpy()
        np.testing.assert_array_equal(vinsgrad_result, torch_result)
    
    def test_argmax_axis2(self):
        """
        Test argmax along axis 2.
        """
        vinsgrad_result = vinsgrad.argmax(self.tensor, axis=2).data
        torch_result = torch.argmax(self.torch_tensor, dim=2).numpy()
        np.testing.assert_array_equal(vinsgrad_result, torch_result)

if __name__ == '__main__':
    unittest.main()
