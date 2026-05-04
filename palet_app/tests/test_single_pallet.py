"""src.core.single_pallet — geometric helpers ve adaptive threshold."""

import random

from django.test import SimpleTestCase

from src.core.single_pallet import (
    MAX_SINGLE_THRESHOLD,
    MIN_SINGLE_THRESHOLD,
    _MIN_BASE_FILL_FOR_SINGLE,
    compute_adaptive_single_threshold,
    compute_max_base_fill,
)


class ComputeMaxBaseFillTests(SimpleTestCase):
    def test_zero_dimension_returns_zero(self):
        fill, orientation, count = compute_max_base_fill(0, 100, 40, 30)
        self.assertEqual(fill, 0.0)
        self.assertEqual(orientation, "0deg")
        self.assertEqual(count, 0)

    def test_negative_dimension_returns_zero(self):
        fill, _, count = compute_max_base_fill(-1, 100, 40, 30)
        self.assertEqual(fill, 0.0)
        self.assertEqual(count, 0)

    def test_perfect_grid_fit(self):
        fill, _, count = compute_max_base_fill(120, 100, 40, 50)
        self.assertEqual(count, 6)
        self.assertAlmostEqual(fill, 1.0, places=4)

    def test_picks_better_orientation(self):
        fill_0, _, _ = compute_max_base_fill(120, 100, 50, 40)
        self.assertGreater(fill_0, 0.0)


class ComputeAdaptiveThresholdTests(SimpleTestCase):
    def test_zero_fill_returns_max(self):
        self.assertEqual(compute_adaptive_single_threshold(0.0), MAX_SINGLE_THRESHOLD)

    def test_negative_fill_returns_max(self):
        self.assertEqual(compute_adaptive_single_threshold(-0.1), MAX_SINGLE_THRESHOLD)

    def test_below_min_base_fill_returns_max(self):
        below = _MIN_BASE_FILL_FOR_SINGLE - 0.05
        self.assertEqual(compute_adaptive_single_threshold(below), MAX_SINGLE_THRESHOLD)

    def test_intermediate_value(self):
        t = compute_adaptive_single_threshold(0.85)
        self.assertGreaterEqual(t, MIN_SINGLE_THRESHOLD)
        self.assertLessEqual(t, MAX_SINGLE_THRESHOLD)

    def test_perfect_fill_clamped(self):
        t = compute_adaptive_single_threshold(1.0)
        self.assertGreaterEqual(t, MIN_SINGLE_THRESHOLD)
        self.assertLessEqual(t, MAX_SINGLE_THRESHOLD)

    def test_random_inputs_always_in_bounds(self):
        rng = random.Random(42)
        for _ in range(200):
            fill = rng.uniform(0, 1.2)
            t = compute_adaptive_single_threshold(fill)
            self.assertGreaterEqual(t, MIN_SINGLE_THRESHOLD)
            self.assertLessEqual(t, MAX_SINGLE_THRESHOLD)
