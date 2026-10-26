import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines import BaselineSuite, CausalFeatures, FeatureConfig, causal_feature_row, naive_predictions


def panel(days=25):
    dates = pd.date_range("2026-01-01", periods=24 * days, freq="h")
    return pd.concat([pd.DataFrame({
        "series_id": sid, "timestamp": dates,
        "fare": 15 + offset + 3 * np.sin(2 * np.pi * dates.hour / 24),
        "count": 20 + dates.hour,
    }) for sid, offset in (("1_2", 0), ("3_4", 5))], ignore_index=True)


class CausalTests(unittest.TestCase):
    def test_no_future_features(self):
        original = panel()
        origin = pd.Timestamp("2026-01-20 12:00")
        changed = original.copy()
        forbidden = changed.timestamp.ge(origin - pd.Timedelta(hours=3))
        changed.loc[forbidden, ["fare", "count"]] = 99999
        one = causal_feature_row(original, "1_2", origin, 6)
        two = causal_feature_row(changed, "1_2", origin, 6)
        pd.testing.assert_series_equal(pd.Series(one), pd.Series(two))

    def test_embargo_exact_boundary(self):
        data = panel()
        origin = pd.Timestamp("2026-01-20 12:00")
        data.loc[data.timestamp.eq(origin - pd.Timedelta(hours=4)), "fare"] = 123
        data.loc[data.timestamp.eq(origin - pd.Timedelta(hours=3)), "fare"] = 999
        row = causal_feature_row(data, "1_2", origin, 1)
        self.assertEqual(row["fare_lag_0"], 123)
        self.assertNotIn("y_true", row)
        self.assertNotIn("target_count", row)

    def test_seasonal_alignment(self):
        cases = CausalFeatures(panel()).cases([pd.Timestamp("2026-01-20 12:00")], [1, 3, 6])
        predictions = naive_predictions(cases)
        for name in ("Daily seasonal", "Weekly seasonal", "Four-week seasonal"):
            np.testing.assert_allclose(predictions[name], cases.y_true, atol=1e-10)

    def test_no_imputed_truth(self):
        data = panel()
        origin = pd.Timestamp("2026-01-20 12:00")
        data.loc[data.timestamp.eq(origin), "fare"] = np.nan
        cases = CausalFeatures(data).cases([origin], [1])
        self.assertTrue(cases.y_true.isna().all())
        self.assertTrue(np.isfinite(naive_predictions(cases)["Persistence"]).all())

    def test_fit_predict_smoke(self):
        suite = BaselineSuite(panel(), horizons=(1, 3), context_hours=672)
        validation, metadata = suite.fit("2026-01-17", "2026-01-17", "2026-01-21", origin_stride=6, quick=True)
        self.assertEqual(len(validation.model.unique()), 6)
        self.assertTrue(metadata["refit_after_validation"])
        self.assertEqual(metadata["training_label_cutoff_exclusive"], "2026-01-16T21:00:00")
        self.assertEqual(metadata["refit"]["Ridge"]["label_cutoff_exclusive"], "2026-01-20T21:00:00")
        self.assertTrue(validation.target_timestamp.lt(pd.Timestamp("2026-01-20T21:00:00")).all())
        self.assertEqual(metadata["training_origin_stride_hours"], 1)
        self.assertEqual(metadata["validation_origin_stride_hours"], 6)
        output = suite.predict(pd.date_range("2026-01-21", periods=4, freq="6h"))
        self.assertEqual(len(output), 2 * 2 * 4 * 6)
        self.assertTrue(np.isfinite(output.prediction).all())
        self.assertTrue((output.last_available_timestamp < output.origin - pd.Timedelta(hours=3)).all())

    def test_local_calendar(self):
        data = panel()
        row = causal_feature_row(data, "1_2", pd.Timestamp("2026-01-20 12:00"), 1)
        self.assertEqual(row["hour"], 7)

    def test_training_order_required(self):
        suite = BaselineSuite(panel())
        with self.assertRaises(ValueError):
            suite.fit("2026-01-17", "2026-01-16", "2026-01-21", quick=True)

    def test_test_outcomes_cannot_change_fit_or_first_prediction(self):
        original = panel()
        changed = original.copy()
        changed.loc[changed.timestamp.ge(pd.Timestamp("2026-01-21")), ["fare", "count"]] = 9999
        a, b = BaselineSuite(original), BaselineSuite(changed)
        val_a, meta_a = a.fit("2026-01-17", "2026-01-17", "2026-01-21", origin_stride=6, quick=True)
        val_b, meta_b = b.fit("2026-01-17", "2026-01-17", "2026-01-21", origin_stride=6, quick=True)
        np.testing.assert_allclose(val_a.prediction, val_b.prediction)
        pred_a = a.predict([pd.Timestamp("2026-01-21")])
        pred_b = b.predict([pd.Timestamp("2026-01-21")])
        np.testing.assert_allclose(pred_a.prediction, pred_b.prediction)

    def test_late_validation_labels_are_not_used(self):
        original = panel()
        changed = original.copy()
        first_test = pd.Timestamp("2026-01-21")
        changed.loc[changed.timestamp.ge(first_test - pd.Timedelta(hours=3)), ["fare", "count"]] = 9999
        a, b = BaselineSuite(original), BaselineSuite(changed)
        val_a, _ = a.fit("2026-01-17", "2026-01-17", first_test, origin_stride=6, quick=True)
        val_b, _ = b.fit("2026-01-17", "2026-01-17", first_test, origin_stride=6, quick=True)
        pd.testing.assert_frame_equal(val_a, val_b)
        np.testing.assert_allclose(a.predict([first_test]).prediction, b.predict([first_test]).prediction)


if __name__ == "__main__":
    unittest.main()
