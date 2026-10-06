import tempfile
import unittest
from pathlib import Path

from neural_network.training_config import TrainingConfig, load_training_config, save_training_config


class TrainingConfigTest(unittest.TestCase):
    def test_training_config_roundtrip(self):
        config = TrainingConfig(
            epochs=3,
            batch_size=2,
            learning_rate=0.01,
            train_test_split=0.2,
            seed=7,
            num_aug=1,
            weight_decay=0.001,
            loss_alpha=0.5,
            loss_beta=2.0,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            save_training_config(config, path)

            self.assertEqual(config, load_training_config(path))

    def test_missing_fields_use_defaults(self):
        config = TrainingConfig.from_dict({"epochs": 5})

        self.assertEqual(5, config.epochs)
        self.assertGreater(config.batch_size, 0)


if __name__ == "__main__":
    unittest.main()
