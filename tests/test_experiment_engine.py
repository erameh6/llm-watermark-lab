import unittest
from backend.app.experiment_engine import _classification, _rate_ci, ATTACKS

class ExperimentEngineTests(unittest.TestCase):
    def test_confusion_metrics(self):
        rows = [
            {"label": 1, "prediction": 1},
            {"label": 1, "prediction": 0},
            {"label": 0, "prediction": 0},
            {"label": 0, "prediction": 1},
        ]
        result = _classification(rows)
        self.assertEqual(result["true_positive"], 1)
        self.assertEqual(result["true_negative"], 1)
        self.assertEqual(result["false_positive"], 1)
        self.assertEqual(result["false_negative"], 1)
        self.assertEqual(result["accuracy"], 0.5)

    def test_wilson_interval_is_informative_for_all_success_or_failure(self):
        lo, hi = _rate_ci([1, 1, 1])
        self.assertLess(lo, 1.0)
        self.assertEqual(hi, 1.0)
        lo, hi = _rate_ci([0, 0, 0])
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.0)

    def test_expected_attacks_are_documented(self):
        self.assertIn("none", ATTACKS)
        self.assertIn("word_deletion", ATTACKS)
        self.assertIn("word_substitution", ATTACKS)
        self.assertIn("lexical_paraphrase_proxy", ATTACKS)

if __name__ == "__main__":
    unittest.main()
