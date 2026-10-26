"""Data preparation checks use synthetic records, never test-period route selection."""
import importlib.util
from collections import Counter
from pathlib import Path
import unittest
import pandas as pd
import numpy as np

spec = importlib.util.spec_from_file_location("prepare_data", Path(__file__).resolve().parents[1] / "prepare_data.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def sample():
    return pd.DataFrame({"hvfhs_license_num": ["HV0003"] * 3,
        "pickup_datetime": pd.to_datetime(["2026-01-01 01:00"]*3),
        "dropoff_datetime": pd.to_datetime(["2026-01-01 01:30"]*3),
        "PULocationID": [79]*3, "DOLocationID": [113]*3,
        "base_passenger_fare": [20., 25., 30.], "trip_miles": [2.]*3,
        "trip_time": [1800]*3, "shared_request_flag": ["N"]*3,
        "shared_match_flag": ["N"]*3, "access_a_ride_flag": ["N"]*3})


class DataTests(unittest.TestCase):
    def test_disjoint_exclusions(self):
        x = sample()
        x.loc[0, "hvfhs_license_num"] = "HV0005"
        x.loc[1, "base_passenger_fare"] = -1
        audit = Counter()
        out = m.clean_batch(x, "2026-01", audit)
        self.assertEqual(len(out), 1)
        self.assertEqual(audit["rows_read"], audit["rows_retained"] + sum(v for k,v in audit.items() if k.startswith("exclude_")))

    def test_long_trip_removed(self):
        x = sample()
        x.loc[0, "dropoff_datetime"] = pd.Timestamp("2026-01-01 05:00")
        self.assertEqual(len(m.clean_batch(x, "2026-01", Counter())), 2)

    def test_threshold_masks_target_not_zero(self):
        x = sample()
        x["timestamp"] = x.pickup_datetime.dt.tz_localize("America/New_York").dt.tz_convert("UTC").dt.tz_localize(None).dt.floor("h")
        x["duration_min"] = 30.
        selected = pd.DataFrame({"series_id": ["79_113"]})
        panel = m.aggregate_selected([x], selected)
        self.assertEqual(len(panel), 2159)
        hour = panel.loc[panel.timestamp.eq("2026-01-01 06:00")].iloc[0]
        self.assertEqual(hour["count"], 3)
        self.assertTrue(np.isnan(hour.fare))

    def test_dst_invalid_source_removed(self):
        x = sample()
        x["pickup_datetime"] = pd.to_datetime(["2026-03-08 01:30", "2026-03-08 02:30", "2026-03-08 03:30"])
        x["dropoff_datetime"] = pd.to_datetime(["2026-03-08 03:30", "2026-03-08 03:30", "2026-03-08 04:00"])
        audit = Counter()
        out = m.clean_batch(x, "2026-03", audit)
        self.assertEqual(len(out), 2)
        self.assertEqual(audit["exclude_invalid_dst_clock_time"], 1)
        self.assertEqual(out.iloc[0].timestamp, pd.Timestamp("2026-03-08 06:00"))

    def test_utc_grid_has_no_dst_gap(self):
        x = sample()
        x["timestamp"] = pd.Timestamp("2026-01-01 06:00")
        x["duration_min"] = 30.
        panel = m.aggregate_selected([x], pd.DataFrame({"series_id": ["79_113"]}))
        self.assertTrue(panel.timestamp.diff().dropna().dt.total_seconds().eq(3600).all())
        local = panel.timestamp.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
        spring = local.dt.strftime("%Y-%m-%d").eq("2026-03-08")
        self.assertEqual(spring.sum(), 23)
        self.assertFalse(panel.loc[spring, "local_hour"].eq(2).any())

    def test_median_uses_records_not_batch_medians(self):
        x = pd.concat([sample()] * 4, ignore_index=True)
        x["timestamp"] = pd.Timestamp("2026-01-01 06:00")
        x["duration_min"] = 30.
        x["base_passenger_fare"] = [10.] * 10 + [100., 200.]
        panel = m.aggregate_selected([x.iloc[:10], x.iloc[10:]], pd.DataFrame({"series_id": ["79_113"]}))
        hour = panel.loc[panel.timestamp.eq("2026-01-01 06:00")].iloc[0]
        self.assertEqual(hour.fare, 10.)
        self.assertEqual(hour["count"], 12)


if __name__ == "__main__":
    unittest.main()
