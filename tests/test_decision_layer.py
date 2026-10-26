import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

spec=importlib.util.spec_from_file_location("decision_layer",Path(__file__).resolve().parents[1]/"decision_layer.py")
m=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=m
spec.loader.exec_module(m)


class DecisionLayerTests(unittest.TestCase):
    def test_fixed_point_is_satisfied(self):
        cfg=m.ResponseConfig()
        for A,S,Q,a in [(50,25,0,1),(150,30,8,1.6),(10,100,0,1),(300,5,30,2)]:
            row=m.solve_response(A,S,Q,a,cfg)
            rhs=min(cfg.max_eta_minutes,cfg.base_eta_minutes+cfg.congestion_gain_minutes*max((Q+row["expected_requests"])/S-cfg.load_threshold,0))
            self.assertAlmostEqual(row["quoted_eta_proxy_minutes"],rhs,places=7)

    def test_price_and_wait_partials_reduce_request_probability(self):
        self.assertLess(m.request_probability(7,1.5),m.request_probability(7,1.))
        self.assertLess(m.request_probability(12,1.),m.request_probability(7,1.))

    def test_queue_priority_and_flow_conservation(self):
        for row in m.response_surface(100,30,12):
            self.assertAlmostEqual(12+row["expected_requests"],row["total_completions"]+row["queue_end"])
            self.assertEqual(row["existing_queue_completions"],12)
            self.assertLessEqual(row["new_rider_completions"],18)
            self.assertLessEqual(row["new_rider_completions"],row["expected_requests"])
            self.assertAlmostEqual(row["new_rider_conversion"],row["new_rider_completions"]/100)

    def test_existing_queue_can_use_all_capacity(self):
        row=m.solve_response(100,20,30,1)
        self.assertEqual(row["new_rider_completions"],0)
        self.assertEqual(row["existing_queue_completions"],20)

    def test_endogenous_wait_does_not_create_price_demand_inversion(self):
        rows=m.response_surface(120,30,5)
        self.assertTrue(np.all(np.diff([r["expected_requests"] for r in rows])<=1e-7))
        self.assertTrue(np.all(np.diff([r["quoted_eta_proxy_minutes"] for r in rows])<=1e-7))
        self.assertTrue(np.all(np.diff([r["new_rider_completions"] for r in rows])<=1e-7))

    def test_feasible_rule_selects_lowest_price_on_tie(self):
        result=m.choose_action(100,20,0,eta_limit=120)
        self.assertTrue(result["feasible"])
        self.assertEqual(result["selected_action"]["price_multiplier"],1)

    def test_infeasibility_is_not_hidden(self):
        result=m.choose_action(500,5,100,eta_limit=10)
        self.assertFalse(result["feasible"])
        self.assertIsNone(result["selected_action"])

    def test_no_capacity_and_clipped_negative_forecasts(self):
        row=m.solve_response(30,0,4,1)
        self.assertEqual(row["total_completions"],0)
        self.assertEqual(row["quoted_eta_proxy_minutes"],120)
        self.assertFalse(m.choose_action(30,0,4)["feasible"])
        clipped=m.solve_response(-4,20,-2,1)
        self.assertEqual(clipped["inputs_clipped_to_zero"],{"potential":-4.,"queue":-2.})
        self.assertEqual(clipped["new_rider_conversion"],0)

    def test_forecast_action_is_evaluated_without_reselection(self):
        forecast=m.choose_action(25,40,0)
        action=forecast["selected_action"]["price_multiplier"]
        realized=m.solve_response(200,15,5,action)
        self.assertEqual(realized["price_multiplier"],action)
        self.assertGreater(realized["quoted_eta_proxy_minutes"],forecast["eta_limit_minutes"])

    def test_invalid_input_fails_explicitly(self):
        with self.assertRaises(ValueError):m.solve_response(np.nan,20,0,1)
        with self.assertRaises(ValueError):m.choose_action(20,20,0,[.9,1.1])
        with self.assertRaises(ValueError):m.response_surface(20,20,0,[])
        with self.assertRaises(ValueError):m.solve_response(20,20,0,1,{"alpha":(-1,1,2)})


if __name__=="__main__":unittest.main()
