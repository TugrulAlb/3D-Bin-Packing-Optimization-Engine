"""palet_app.models — Optimization, Palet, Urun davranış testleri."""

import json

from django.test import TestCase

from palet_app.models import Optimization, Palet, Urun


def _make_optimization(**overrides):
    defaults = dict(
        container_length=120,
        container_width=100,
        container_height=180,
        container_weight=1250,
        algoritma="greedy",
        islem_durumu='{"current_step": 0, "total_steps": 5, "messages": []}',
    )
    defaults.update(overrides)
    return Optimization.objects.create(**defaults)


class OptimizationModelTests(TestCase):
    def test_tamamla_sets_bitis_zamani_and_flag(self):
        opt = _make_optimization()
        self.assertFalse(opt.tamamlandi)
        self.assertIsNone(opt.bitis_zamani)
        opt.tamamla()
        opt.refresh_from_db()
        self.assertTrue(opt.tamamlandi)
        self.assertIsNotNone(opt.bitis_zamani)

    def test_islem_adimi_ekle_clamps_to_total_steps(self):
        opt = _make_optimization(
            islem_durumu='{"current_step": 0, "total_steps": 3, "messages": []}'
        )
        for i in range(5):
            opt.islem_adimi_ekle(f"step {i}")
        durum = opt.get_islem_durumu()
        self.assertEqual(durum["current_step"], 3)
        self.assertEqual(len(durum["messages"]), 5)

    def test_set_phase_ignores_unknown_phase(self):
        opt = _make_optimization()
        before = opt.islem_durumu
        opt.set_phase("unknown_phase_xyz")
        opt.refresh_from_db()
        self.assertEqual(opt.islem_durumu, before)

    def test_set_phase_persists_phase_and_expected_sec(self):
        opt = _make_optimization()
        opt.set_phase("mix", expected_sec=42.0)
        durum = opt.get_islem_durumu()
        self.assertEqual(durum["phase"], "mix")
        self.assertEqual(durum["phase_expected_sec"], 42.0)
        self.assertIn("phase_start", durum)

    def test_get_islem_durumu_round_trip(self):
        opt = _make_optimization()
        opt.set_phase("mix", expected_sec=10)
        opt.refresh_from_db()
        durum = opt.get_islem_durumu()
        self.assertIsInstance(durum, dict)
        self.assertEqual(durum["phase"], "mix")


class PaletModelTests(TestCase):
    def setUp(self):
        self.opt = _make_optimization()

    def test_dimension_fallbacks_when_custom_fields_none(self):
        p = Palet.objects.create(
            optimization=self.opt, palet_id=1, palet_turu="single",
        )
        self.assertEqual(p.en, 100.0)
        self.assertEqual(p.boy, 120.0)
        self.assertEqual(p.max_yukseklik, 180.0)
        self.assertEqual(p.max_agirlik, 1250.0)

    def test_dimension_uses_custom_when_provided(self):
        p = Palet.objects.create(
            optimization=self.opt, palet_id=2, palet_turu="mix",
            custom_en=80, custom_boy=120, custom_max_yukseklik=200,
            custom_max_agirlik=2000,
        )
        self.assertEqual(p.en, 80)
        self.assertEqual(p.boy, 120)
        self.assertEqual(p.max_yukseklik, 200)
        self.assertEqual(p.max_agirlik, 2000)

    def test_hacim_returns_product_of_dimensions(self):
        p = Palet.objects.create(
            optimization=self.opt, palet_id=3, palet_turu="single",
            custom_en=100, custom_boy=120, custom_max_yukseklik=180,
        )
        self.assertEqual(p.hacim(), 100 * 120 * 180)

    def test_doluluk_orani_percentage_calculation(self):
        p = Palet.objects.create(
            optimization=self.opt, palet_id=4, palet_turu="single",
            custom_en=100, custom_boy=120, custom_max_yukseklik=180,
            kullanilan_hacim=540_000,
        )
        self.assertAlmostEqual(p.doluluk_orani(), 25.0, places=2)

    def test_json_to_dict_parses_string(self):
        p = Palet.objects.create(
            optimization=self.opt, palet_id=5, palet_turu="single",
        )
        self.assertEqual(p.json_to_dict('{"a": 1}'), {"a": 1})

    def test_json_to_dict_passes_dict_through(self):
        p = Palet.objects.create(
            optimization=self.opt, palet_id=6, palet_turu="single",
        )
        self.assertEqual(p.json_to_dict({"x": 9}), {"x": 9})


class UrunModelTests(TestCase):
    def test_str_repr_contains_kod_and_ad(self):
        u = Urun.objects.create(
            urun_kodu="X-1", urun_adi="Test",
            boy=10, en=10, yukseklik=10, agirlik=1, mukavemet=100,
        )
        s = str(u)
        self.assertIn("X-1", s)
        self.assertIn("Test", s)
