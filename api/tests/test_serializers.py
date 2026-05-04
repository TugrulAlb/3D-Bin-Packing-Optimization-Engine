"""api.serializers — _palet_to_dict ve OptimizeRequestSerializer iç davranış."""

from django.test import TestCase

from api.serializers import OptimizeRequestSerializer, serialize_paletler
from palet_app.models import Palet, Urun

from ._helpers import make_optimization


class OptimizeRequestSerializerTests(TestCase):
    def _payload(self, **overrides):
        base = {
            "container": {"length": 120, "width": 100, "height": 180, "weight": 1250},
            "details": [{
                "product": {
                    "code": "X", "package_length": 40, "package_width": 30,
                    "package_height": 25, "package_weight": 10,
                },
                "package_quantity": 1,
            }],
            "algorithm": "greedy",
        }
        base.update(overrides)
        return base

    def test_default_algorithm_is_greedy(self):
        ser = OptimizeRequestSerializer(data=self._payload(algorithm=None))
        ser.initial_data.pop("algorithm", None)
        self.assertTrue(ser.is_valid(), ser.errors)
        self.assertEqual(ser.validated_data["algorithm"], "greedy")

    def test_default_ga_mode_is_balanced(self):
        ser = OptimizeRequestSerializer(data=self._payload())
        self.assertTrue(ser.is_valid(), ser.errors)
        self.assertEqual(ser.validated_data["ga_mode"], "balanced")

    def test_invalid_algorithm_rejected(self):
        ser = OptimizeRequestSerializer(data=self._payload(algorithm="foo"))
        self.assertFalse(ser.is_valid())
        self.assertIn("algorithm", ser.errors)

    def test_invalid_ga_mode_rejected(self):
        ser = OptimizeRequestSerializer(data=self._payload(ga_mode="extreme"))
        self.assertFalse(ser.is_valid())
        self.assertIn("ga_mode", ser.errors)

    def test_unit_dims_used_for_size_check_when_pq_missing(self):
        ser = OptimizeRequestSerializer(data={
            "container": {"length": 120, "width": 100, "height": 180, "weight": 1250},
            "details": [{
                "product": {
                    "code": "Y",
                    "unit_length": 999, "unit_width": 5,
                    "unit_height": 5, "unit_weight": 0.1,
                },
                "package_quantity": None,
                "quantity": 5,
            }],
        })
        self.assertFalse(ser.is_valid())


class SerializePaletlerTests(TestCase):
    def test_empty_paletler_returns_empty_list(self):
        self.assertEqual(serialize_paletler([]), [])

    def test_serializes_palet_with_items(self):
        opt = make_optimization(tamamlandi=True)
        urun = Urun.objects.create(
            urun_kodu="A", urun_adi="A",
            boy=40, en=30, yukseklik=25, agirlik=10, mukavemet=200,
        )
        palet = Palet.objects.create(
            optimization=opt, palet_id=1, palet_turu="single",
            custom_en=100, custom_boy=120, custom_max_yukseklik=180,
            custom_max_agirlik=1250,
            kullanilan_hacim=30000,
            urun_konumlari={str(urun.id): [0, 0, 0]},
            urun_boyutlari={str(urun.id): [40, 30, 25]},
        )
        result = serialize_paletler([palet])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["urun_sayisi"], 1)
        self.assertEqual(result[0]["dimensions"]["boy_cm"], 120.0)

    def test_skips_invalid_urun_id_silently(self):
        opt = make_optimization(tamamlandi=True)
        palet = Palet.objects.create(
            optimization=opt, palet_id=1, palet_turu="single",
            custom_en=100, custom_boy=120, custom_max_yukseklik=180,
            custom_max_agirlik=1250,
            urun_konumlari={"99999": [0, 0, 0]},
            urun_boyutlari={"99999": [10, 10, 10]},
        )
        result = serialize_paletler([palet])
        self.assertEqual(result[0]["urun_sayisi"], 0)
