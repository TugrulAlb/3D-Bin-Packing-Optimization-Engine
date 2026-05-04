"""normalize_progress + phase_progress + estimate_mix_sec testleri."""

from django.test import SimpleTestCase

from palet_app.workers.progress import (
    estimate_mix_sec,
    normalize_progress,
    phase_progress,
)


class NormalizeProgressTests(SimpleTestCase):
    def test_zero_step_returns_zero_pct(self):
        cur, tot, pct = normalize_progress(0, 5)
        self.assertEqual((cur, tot, pct), (0, 5, 0))

    def test_clamps_to_95_when_not_completed(self):
        _, _, pct = normalize_progress(5, 5, completed=False)
        self.assertEqual(pct, 95)

    def test_completed_returns_100(self):
        _, _, pct = normalize_progress(5, 5, completed=True)
        self.assertEqual(pct, 100)

    def test_negative_step_clamped_to_zero(self):
        cur, _, _ = normalize_progress(-1, 5)
        self.assertEqual(cur, 0)

    def test_total_steps_zero_clamps_to_one(self):
        cur, tot, pct = normalize_progress(1, 0)
        self.assertEqual(tot, 1)
        self.assertEqual(cur, 1)
        self.assertEqual(pct, 95)


class PhaseProgressTests(SimpleTestCase):
    def test_completed_returns_100(self):
        pct, label = phase_progress({"phase": "mix"}, completed=True)
        self.assertEqual(pct, 100)
        self.assertEqual(label, "Tamamlandı")

    def test_unknown_phase_returns_last_pct(self):
        pct, _ = phase_progress({"phase": "bilinmeyen", "last_pct": 42})
        self.assertEqual(pct, 42)

    def test_no_phase_returns_default_label(self):
        pct, label = phase_progress({})
        self.assertEqual(pct, 0)
        self.assertEqual(label, "Hazırlanıyor")

    def test_phase_progress_within_range(self):
        clock = [1000.0]

        def fake_time():
            clock[0] += 1.0
            return clock[0]

        durum = {
            "phase": "mix",
            "phase_start": 1000.0,
            "phase_expected_sec": 60.0,
            "last_pct": 0,
        }
        pct, label = phase_progress(durum, time_func=fake_time)
        self.assertEqual(label, "Mix palet optimizasyonu")
        self.assertGreaterEqual(pct, 25)
        self.assertLess(pct, 75)

    def test_monotonic_via_last_pct(self):
        durum = {
            "phase": "single",
            "phase_start": 1000.0,
            "phase_expected_sec": 60.0,
            "last_pct": 80,
        }
        pct, _ = phase_progress(durum, time_func=lambda: 1000.0)
        self.assertEqual(pct, 80)


class EstimateMixSecTests(SimpleTestCase):
    def test_greedy_lower_bound(self):
        self.assertGreaterEqual(estimate_mix_sec("greedy", 0), 3.0)

    def test_greedy_scales_with_count(self):
        self.assertGreater(estimate_mix_sec("greedy", 1000), estimate_mix_sec("greedy", 10))

    def test_de_slower_than_greedy(self):
        self.assertGreater(
            estimate_mix_sec("differential_evolution", 100),
            estimate_mix_sec("greedy", 100),
        )

    def test_unknown_algorithm_falls_back_to_genetic_branch(self):
        self.assertGreaterEqual(estimate_mix_sec("anything-else", 100), 10.0)
