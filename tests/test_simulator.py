import importlib.util
from pathlib import Path
from copy import deepcopy
import unittest

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("simulator", Path(__file__).resolve().parents[1] / "simulator.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class SimulatorTests(unittest.TestCase):
    def setUp(self):
        self.parameters = m.sample_parameters(np.random.default_rng(11),
            {"median_fare_usd": 13., "median_positive_completed_count": 20.})
        self.exo = m.make_exogenous(self.parameters, 42, length=96)

    def test_price_and_eta_reduce_choice_all_else_fixed(self):
        p = self.parameters
        f = lambda price,eta: m.choice_probability(price,eta,p["base_anchor"],p["base_eta"],p["alpha"],p["beta"],p["intercept"])
        self.assertTrue(np.all(f(2*p["base_anchor"],p["base_eta"]) < f(p["base_anchor"],p["base_eta"])))
        self.assertTrue(np.all(f(p["base_anchor"],p["base_eta"]+5) < f(p["base_anchor"],p["base_eta"])))

    def test_flow_conservation_and_nonnegativity(self):
        for policy in m.POLICIES:
            out = m.simulate_path(self.exo,self.parameters,policy)
            np.testing.assert_array_equal(out["queue_start"]+out["requested"],out["abandoned"]+out["count"]+out["queue_end"])
            self.assertEqual(out["requested"].sum(),out["abandoned"].sum()+out["count"].sum()+out["queue_end"][-1])
            self.assertTrue(np.all(out["count"] <= out["service_budget"]))
            self.assertTrue(np.all(out["requested"] <= out["potential"]))
            for key in ("count","requested","abandoned","queue_end","gross_revenue"):
                self.assertTrue(np.all(out[key] >= 0))
            self.assertTrue(np.isnan(out["fare"][out["count"]<10]).all())

    def test_future_mutation_cannot_change_past(self):
        exo = deepcopy(self.exo)
        # Change only future observable capacity and future realized demand.
        exo["capacity"][48:] *= .5
        exo["forecast_potential"][48:] *= 2
        exo["potential"][48:] //= 2
        for policy in m.POLICIES:
            a = m.simulate_path(self.exo,self.parameters,policy)
            b = m.simulate_path(exo,self.parameters,policy)
            for key in ("fare","count","posted_fare","quoted_eta","queue_end"):
                np.testing.assert_allclose(a[key][:48],b[key][:48],equal_nan=True)

    def test_capacity_not_changed_by_rider_price(self):
        paths,_ = m.policy_comparison(self.exo,self.parameters)
        for policy in m.POLICIES:
            np.testing.assert_array_equal(paths[policy]["capacity"],self.exo["capacity"])

    def test_heldout_values_do_not_change_scale_calibration(self):
        frame = pd.DataFrame({"timestamp":pd.date_range("2026-02-01",periods=24*30,freq="h"),
                              "fare":13.,"count":20})
        changed = frame.copy()
        changed.loc[changed.timestamp>=m.CALIBRATION_CUTOFF,["fare","count"]] = 1e9
        self.assertEqual(m.calibrate_scales(frame),m.calibrate_scales(changed))

    def test_cohort_median_and_locked_fares(self):
        self.assertEqual(m.cohort_median([(3,10.),(1,40.)]),10.)
        self.assertEqual(m.cohort_median([(2,10.),(2,40.)]),25.)
        queue = m.deque([[3,10.],[2,40.]])
        self.assertEqual(m._take_fifo(queue,4),[(3,10.),(1,40.)])
        self.assertEqual(list(queue),[[1,40.]])

    def test_reproducible_corpus(self):
        scales = {"median_fare_usd":13.,"median_positive_completed_count":20.}
        a,ma = m._corpus(scales,123,2,48)
        b,mb = m._corpus(scales,123,2,48)
        self.assertEqual(ma,mb)
        for key in a:
            np.testing.assert_array_equal(a[key],b[key])


if __name__ == "__main__":
    unittest.main()
