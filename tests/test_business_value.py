import importlib.util
from dataclasses import replace
from pathlib import Path
import sys
import unittest

spec=importlib.util.spec_from_file_location("business_value",Path(__file__).resolve().parents[1]/"business_value.py")
m=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=m
spec.loader.exec_module(m)


class BusinessValueTests(unittest.TestCase):
    def test_scale_and_percentage_point_units(self):
        r=m.scenario(.05)
        self.assertEqual(r["annualized_mobility_gross_bookings_scale_usd"],115_952_000_000)
        self.assertEqual(r["eligible_annualized_gross_bookings_scale_usd"],23_190_400_000)
        self.assertAlmostEqual(r["conversion_after_fraction"],.6005)
        self.assertEqual(r["additional_completions_per_10000_quotes"],5)
        self.assertAlmostEqual(r["incremental_gross_bookings_usd"],19_325_333.333333,places=3)

    def test_zero_and_negative_lift_are_losses(self):
        self.assertEqual(m.scenario(0)["net_incremental_contribution_after_program_cost_usd"],-1_000_000)
        self.assertLess(m.scenario(-.05)["net_incremental_contribution_after_program_cost_usd"],-1_000_000)

    def test_default_plus_005pp_net(self):
        self.assertAlmostEqual(m.scenario(.05)["net_incremental_contribution_after_program_cost_usd"],932_533.333333,places=3)

    def test_break_even_closes_for_fare_changes(self):
        for fare in [-.001,-.0001,0,.0001]:
            a=replace(m.Assumptions(),average_fare_change_relative=fare)
            r=m.scenario(m.break_even_delta_pp(a),a)
            self.assertAlmostEqual(r["net_incremental_contribution_after_program_cost_usd"],0,places=4)

    def test_two_million_cost_makes_example_negative(self):
        r=m.scenario(.05,replace(m.Assumptions(),annual_all_in_program_cost_usd=2_000_000))
        self.assertLess(r["net_incremental_contribution_after_program_cost_usd"],0)

    def test_discount_existing_rides_can_offset_gain(self):
        r=m.scenario(.05,replace(m.Assumptions(),average_fare_change_relative=-.001))
        self.assertLess(r["incremental_gross_bookings_usd"],0)

    def test_price_channel_not_mistaken_for_volume_margin(self):
        a=replace(m.Assumptions(),average_fare_change_relative=-.0001,price_retention_factor=1.)
        r=m.scenario(.05,a)
        G=23_190_400_000
        volume=.0005/.60
        self.assertAlmostEqual(r["incremental_contribution_before_program_cost_usd"],G*.1*volume+G*(1+volume)*(-.0001),places=4)

    def test_coupon_cost_is_separate_from_reported_bookings(self):
        original=m.scenario(.05)
        coupon=m.scenario(.05,replace(m.Assumptions(),extra_subsidy_usd=100_000))
        self.assertEqual(original["incremental_gross_bookings_usd"],coupon["incremental_gross_bookings_usd"])
        self.assertAlmostEqual(original["net_incremental_contribution_after_program_cost_usd"]-coupon["net_incremental_contribution_after_program_cost_usd"],100_000)

    def test_nonpositive_break_even_slope_rejected(self):
        with self.assertRaises(ValueError):m.break_even_delta_pp(replace(m.Assumptions(),average_fare_change_relative=-.20))

    def test_compute_no_extra_thread_multiplication(self):
        x=m.compute_illustration()
        self.assertEqual(x["forecasts_per_day"],2_880_000)
        self.assertEqual(x["forecasts_per_year"],1_051_200_000)
        self.assertAlmostEqual(x["worker_hours_per_year"],1_051_200_000*.1276987327917202/3600)
        self.assertAlmostEqual(x["illustrative_annual_model_compute_usd"],x["worker_hours_per_year"]*.2)

    def test_invalid_assumptions_fail(self):
        with self.assertRaises(ValueError):m.scenario(50)
        with self.assertRaises(ValueError):m.scenario(0,replace(m.Assumptions(),baseline_quote_to_completed_conversion=0))
        with self.assertRaises(ValueError):m.break_even_delta_pp(replace(m.Assumptions(),incremental_contribution_rate=0))


if __name__=="__main__":unittest.main()
