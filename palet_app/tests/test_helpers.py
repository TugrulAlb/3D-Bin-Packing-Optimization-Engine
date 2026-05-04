"""src.utils.helpers + src.models — küçük saf fonksiyon testleri."""

from django.test import SimpleTestCase

from src.models import PaletConfig, UrunData
from src.utils.helpers import group_products_smart, urun_hacmi


def _ud(code="A1", boy=10, en=20, yukseklik=30, agirlik=1.0, urun_id=1):
    return UrunData(
        urun_id=urun_id, code=code,
        boy=boy, en=en, yukseklik=yukseklik, agirlik=agirlik,
    )


class UrunHacmiTests(SimpleTestCase):
    def test_returns_product_of_three_dims(self):
        u = _ud(boy=10, en=20, yukseklik=30)
        self.assertEqual(urun_hacmi(u), 6000.0)


class GroupProductsSmartTests(SimpleTestCase):
    def test_empty_returns_empty(self):
        self.assertEqual(group_products_smart([]), {})

    def test_groups_identical_products(self):
        a = _ud(code="A", urun_id=1)
        b = _ud(code="A", urun_id=2)
        groups = group_products_smart([a, b])
        self.assertEqual(len(groups), 1)
        only = next(iter(groups.values()))
        self.assertEqual(len(only), 2)

    def test_different_dimensions_separate_groups(self):
        a = _ud(code="A", boy=10, urun_id=1)
        b = _ud(code="A", boy=20, urun_id=2)
        groups = group_products_smart([a, b])
        self.assertEqual(len(groups), 2)


class PaletConfigTests(SimpleTestCase):
    def test_volume_property(self):
        cfg = PaletConfig(120, 100, 180, 1250)
        self.assertEqual(cfg.volume, 120 * 100 * 180)

    def test_repr_contains_dims(self):
        cfg = PaletConfig(120, 100, 180, 1250)
        s = repr(cfg)
        self.assertIn("120", s)
        self.assertIn("1250", s)


class UrunDataTests(SimpleTestCase):
    def test_repr_format(self):
        u = _ud(code="X", boy=10, en=20, yukseklik=30)
        s = repr(u)
        self.assertIn("X", s)
        self.assertIn("10.0", s)
