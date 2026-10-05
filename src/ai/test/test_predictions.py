import json
import tempfile
import unittest
from pathlib import Path

from src.ai.model_metadata import MODEL_METADATA, load_model_metadata, write_model_metadata
from src.ai.predictions import decode_predictions


class PredictionsTest(unittest.TestCase):
    def test_decode_predictions_filters_by_confidence_and_radius(self):
        shots = decode_predictions(
            [
                0.9, 0.1, 15.0,
                0.4, 0.2, 20.0,
                0.8, 1.2, 30.0,
            ],
            max_shots=3,
            confidence_threshold=0.5,
        )

        self.assertEqual(1, len(shots))
        self.assertEqual(0.9, shots[0].confidence)
        self.assertEqual(0.1, shots[0].radius_norm)
        self.assertEqual(15.0, shots[0].angle_deg)

    def test_decode_predictions_rejects_short_output(self):
        with self.assertRaisesRegex(ValueError, "expected at least 3 output values"):
            decode_predictions([1.0, 0.2], max_shots=1, confidence_threshold=0.5)

    def test_model_metadata_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_model_metadata(tmp)

            self.assertEqual(MODEL_METADATA, load_model_metadata(Path(tmp) / "model_metadata.json"))

    def test_android_asset_matches_python_metadata(self):
        android_metadata = Path(
            "AndroidApp/ArcheryHelper/app/src/main/assets/model_metadata.json"
        )
        data = json.loads(android_metadata.read_text(encoding="utf-8"))

        self.assertEqual(MODEL_METADATA.to_dict(), data)


if __name__ == "__main__":
    unittest.main()
