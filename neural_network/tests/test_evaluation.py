import unittest

from neural_network.contracts import ShotAnnotation
from neural_network.evaluation import evaluate_detections


class EvaluationTest(unittest.TestCase):
    def test_evaluate_detections_counts_matches_and_errors(self):
        report = evaluate_detections(
            predictions=[
                ShotAnnotation(0.0, 0.0),
                ShotAnnotation(0.5, 0.5),
                ShotAnnotation(-0.5, -0.5),
            ],
            targets=[
                ShotAnnotation(0.01, 0.0),
                ShotAnnotation(0.49, 0.5),
            ],
            match_threshold=0.05,
        )

        self.assertEqual(2, report.true_positive)
        self.assertEqual(1, report.false_positive)
        self.assertEqual(0, report.false_negative)
        self.assertAlmostEqual(2 / 3, report.precision)
        self.assertAlmostEqual(1.0, report.recall)
        self.assertAlmostEqual(0.01, report.localization_mae)

    def test_evaluate_detections_counts_false_negative(self):
        report = evaluate_detections(
            predictions=[ShotAnnotation(0.0, 0.0)],
            targets=[ShotAnnotation(0.0, 0.0), ShotAnnotation(0.5, 0.5)],
            match_threshold=0.05,
        )

        self.assertEqual(1, report.true_positive)
        self.assertEqual(0, report.false_positive)
        self.assertEqual(1, report.false_negative)
        self.assertAlmostEqual(1.0, report.precision)
        self.assertAlmostEqual(0.5, report.recall)


if __name__ == "__main__":
    unittest.main()
