import unittest
import os

import torch
import numpy as np

from slog import utils


class TestUtils(unittest.TestCase):

    def test_convert_tensor_to_numpy(self):
        x = torch.zeros((5, 5, 5))
        numpy_x = utils.convert_tensor_to_numpy(x)
        self.assertTrue(isinstance(numpy_x, np.ndarray))

    def test_convert_tensor_to_cv(self):
        x = torch.zeros((5, 5, 5))
        cv2_x = utils.convert_tensor_to_cv2(x)
        self.assertEqual(cv2_x.dtype, np.uint8)

    def test_merge_dictionaries(self):
        dictionary_a = {"sereact": 123}
        dictionary_b = {"ai": 345}

        merged_dictionary = utils.merge_dictionaries([dictionary_a, dictionary_b])

        self.assertTrue("sereact" in merged_dictionary.keys())
        self.assertTrue("ai" in merged_dictionary.keys())
        self.assertFalse("kanishk" in merged_dictionary.keys())

    def test_initialize_config_file(self):
        dictionary = utils.initialize_config_file('tests/data/don-config.yaml')
        self.assertTrue(type(dictionary), dict)

    def test_dump_as_json(self):
        dictionary = {"sereact": 123}
        dump_path = '/tmp'
        file_name = 'test_json_dump.json'
        utils.dump_as_json(dump_path, file_name, dictionary)
        self.assertTrue(os.path.isfile(os.path.abspath(os.path.join(dump_path, file_name))))

    def test_dump_as_image(self):
        x = torch.ones((256, 256, 3))
        dump_path = '/tmp'
        file_name = 'test_image_dump.png'
        utils.dump_as_image(dump_path, file_name, utils.convert_tensor_to_cv2(x))
        self.assertTrue(os.path.isfile(os.path.abspath(os.path.join(dump_path, file_name))))


if __name__ == '__main__':
    unittest.main()
