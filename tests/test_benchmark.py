import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark import make_chronos_input, calendar_features, common_support, calibrate_baseline_intervals, summarize


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads((ROOT / "protocol.json").read_text())
        self.times = pd.date_range("2026-01-01 05:00", periods=1100, freq="h")
        self.series = pd.DataFrame({"fare": np.arange(1100, dtype=float), "count": 20}, index=self.times)
        self.origin = self.times[1000]

    def test_chronos_embargo_and_horizon(self):
        item = make_chronos_input(self.series, self.origin, self.protocol, True)
        self.assertEqual(item["target"].shape, (1, 672))
        self.assertEqual(item["target"][0, -1], 996)
        self.assertEqual(len(item["future_covariates"]["hour_sin"]), 9)
        self.assertNotIn("served_count", item["future_covariates"])
        self.assertEqual(self.origin - pd.Timedelta(hours=3) + pd.Timedelta(hours=3), self.origin)

    def test_future_values_cannot_change_chronos_inputs(self):
        original = make_chronos_input(self.series, self.origin, self.protocol, True)
        changed = self.series.copy()
        changed.loc[changed.index >= self.origin - pd.Timedelta(hours=3), :] = 999999
        mutated = make_chronos_input(changed, self.origin, self.protocol, True)
        for field in ["target"]:
            np.testing.assert_array_equal(original[field], mutated[field])
        for category in ["past_covariates", "future_covariates"]:
            for key in original[category]:
                np.testing.assert_array_equal(original[category][key], mutated[category][key])

    def test_calendar_respects_dst(self):
        times = pd.DatetimeIndex(["2026-03-08 06:00", "2026-03-08 07:00"])
        features = calendar_features(times)
        np.testing.assert_allclose(features["hour_sin"], np.sin(2*np.pi*np.array([1, 3])/24), rtol=1e-6)

    def test_missing_labels_common_support(self):
        rows = []
        for sid, actual in [("A", 20.), ("B", np.nan)]:
            for model in ["Persistence", "Chronos-2"]:
                rows.append({"origin": self.origin, "series_id": sid, "horizon": 1,
                    "target_timestamp": self.origin, "model": model, "actual": actual, "prediction": 18.})
        good, audit = common_support(pd.DataFrame(rows))
        self.assertEqual(len(good), 2)
        self.assertEqual(audit["excluded_route_origin_horizons"], 1)

    def test_metrics_hand_calculation(self):
        frame = pd.DataFrame({"model": ["test"]*2, "horizon": [1, 1], "actual": [10., 20.],
            "prediction": [12., 17.], "q10": [8., 15.], "q90": [14., 19.],
            "high_fare": [False, True], "interval_method": ["test"]*2})
        result = [x for x in summarize(frame) if x["horizon"] == "all"][0]
        self.assertEqual(result["mae"], 2.5)
        self.assertAlmostEqual(result["rmse"], np.sqrt(6.5))
        self.assertEqual(result["coverage_80"], .5)
        self.assertEqual(result["high_fare_mae"], 3.)


if __name__ == "__main__":
    unittest.main()
