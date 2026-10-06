import csv
import unittest
from pathlib import Path

from neural_network.archery_ml.scoring import ImpactPoint, score
from neural_network.archery_ml.targets import (
    TargetConfig,
    TargetFormat,
    TargetTemplate,
    TenRingMode,
    TripleLayout,
)


FIXTURES_PATH = Path("app/app/src/test/resources/scoring_v2.csv")


class ScoringV2Test(unittest.TestCase):
    def test_shared_scoring_fixtures(self):
        with FIXTURES_PATH.open(encoding="utf-8", newline="") as stream:
            cases = list(csv.DictReader(stream))

        for case in cases:
            with self.subTest(case=case["name"]):
                config = TargetConfig(
                    format=TargetFormat(case["format"]),
                    minimum_scoring_zone=int(case["minimum_zone"]),
                    ten_ring_mode=TenRingMode(case["ten_mode"]),
                    face_diameter_mm=int(case["face_diameter_mm"]),
                    triple_layout=None if not case["layout"] else TripleLayout(case["layout"]),
                )
                result = score(
                    TargetTemplate.from_config(config),
                    ImpactPoint(
                        x_norm=float(case["x_norm"]),
                        y_norm=float(case["y_norm"]),
                        face_index=int(case["face_index"]),
                    ),
                    arrow_diameter_mm=None if not case["arrow_diameter_mm"] else float(case["arrow_diameter_mm"]),
                    localization_error=float(case["localization_error"]),
                )

                self.assertEqual(int(case["score"]), result.final_candidate)
                self.assertEqual(case["is_x"] == "true", result.is_x)
                self.assertEqual(case["line_call"] == "true", result.line_call)

    def test_invalid_target_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            TargetConfig(
                format=TargetFormat.TRIPLE,
                minimum_scoring_zone=6,
                ten_ring_mode=TenRingMode.RECURVE,
                face_diameter_mm=400,
            )


if __name__ == "__main__":
    unittest.main()
