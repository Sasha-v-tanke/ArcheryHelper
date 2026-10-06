import unittest

from neural_network.splits import expanded_sample_indices


class SplitsTest(unittest.TestCase):
    def test_train_indices_include_augmentation_blocks(self):
        indices = expanded_sample_indices([0, 2], base_len=3, num_aug=2, include_augmented=True)

        self.assertEqual([0, 2, 3, 5, 6, 8], indices)

    def test_eval_indices_use_original_samples_only(self):
        indices = expanded_sample_indices([1, 2], base_len=3, num_aug=2, include_augmented=False)

        self.assertEqual([1, 2], indices)


if __name__ == "__main__":
    unittest.main()
