"""parse_optimization_payload regression ve edge-case testleri."""

import json
import os

from django.test import SimpleTestCase

from palet_app.services import parse_optimization_payload


SAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "samples",
)


def _detail(pq=1, qty=10, unit_id="ADET", **product_overrides):
    product = {
        "code": "TEST",
        "package_length": 40,
        "package_width": 30,
        "package_height": 25,
        "package_weight": 10,
        "package_max_stack_weight": 200,
        "unit_length": 11,
        "unit_width": 10,
        "unit_height": 12,
        "unit_weight": 1.0,
    }
    product.update(product_overrides)
    return {
        "product": product,
        "package_quantity": pq,
        "quantity": qty,
        "unit_id": unit_id,
    }


def _payload(details, container=None):
    return {
        "container": container or {"length": 120, "width": 100, "height": 180, "weight": 1250},
        "details": details,
    }


class ParserHappyPathTests(SimpleTestCase):
    def test_pq_3_creates_3_records_with_package_dims(self):
        urunler, _ = parse_optimization_payload(_payload([_detail(pq=3)]))
        self.assertEqual(len(urunler), 3)
        self.assertEqual(urunler[0]["boy"], 40)
        self.assertEqual(urunler[0]["agirlik"], 10)

    def test_pq_none_unit_adet_quantity_5_creates_5_records(self):
        urunler, _ = parse_optimization_payload(
            _payload([_detail(pq=None, qty=5, unit_id="ADET")])
        )
        self.assertEqual(len(urunler), 5)
        self.assertEqual(urunler[0]["boy"], 11)

    def test_pq_zero_falls_back_to_quantity_path(self):
        urunler, _ = parse_optimization_payload(
            _payload([_detail(pq=0, qty=4, unit_id="ADET")])
        )
        self.assertEqual(len(urunler), 4)
        self.assertEqual(urunler[0]["boy"], 11)

    def test_zero_mukavemet_defaults_to_100000(self):
        urunler, _ = parse_optimization_payload(
            _payload([_detail(pq=1, package_max_stack_weight=0)])
        )
        self.assertEqual(urunler[0]["mukavemet"], 100000)

    def test_container_passes_through_with_palet_id(self):
        payload = _payload([_detail(pq=1)])
        payload["id"] = 42
        _, container = parse_optimization_payload(payload)
        self.assertEqual(container["palet_id"], 42)
        self.assertEqual(container["length"], 120)


class ParserValidationTests(SimpleTestCase):
    def test_raises_when_payload_is_none(self):
        with self.assertRaises(ValueError):
            parse_optimization_payload(None)

    def test_raises_when_payload_is_empty_dict(self):
        with self.assertRaises(ValueError):
            parse_optimization_payload({})

    def test_raises_when_payload_is_empty_list(self):
        with self.assertRaises(ValueError):
            parse_optimization_payload([])

    def test_raises_when_details_is_empty_list(self):
        with self.assertRaises(ValueError):
            parse_optimization_payload(_payload([]))

    def test_pq_none_qty_zero_raises(self):
        with self.assertRaises(ValueError):
            parse_optimization_payload(
                _payload([_detail(pq=None, qty=0)])
            )

    def test_pq_above_limit_raises(self):
        with self.assertRaises(ValueError):
            parse_optimization_payload(
                _payload([_detail(pq=999_999_999)])
            )

    def test_details_above_max_rows_raises(self):
        details = [_detail(pq=1) for _ in range(5001)]
        with self.assertRaises(ValueError):
            parse_optimization_payload(_payload(details))

    def test_total_expansion_overflow_raises(self):
        details = [_detail(pq=200_000), _detail(pq=350_000)]
        with self.assertRaises(ValueError):
            parse_optimization_payload(_payload(details))

    def test_non_dict_details_item_raises(self):
        with self.assertRaises(ValueError):
            parse_optimization_payload(_payload(["not-a-dict"]))


class ParserLegacyTests(SimpleTestCase):
    def test_legacy_list_payload_passes_with_required_fields(self):
        legacy = [{
            "urun_kodu": "X1",
            "urun_adi": "X1",
            "boy": 10,
            "en": 10,
            "yukseklik": 10,
            "agirlik": 1,
        }]
        urunler, container = parse_optimization_payload(legacy)
        self.assertEqual(len(urunler), 1)
        self.assertEqual(container, {})

    def test_legacy_list_missing_required_field_raises(self):
        legacy = [{"urun_kodu": "X1"}]
        with self.assertRaises(ValueError):
            parse_optimization_payload(legacy)


class ParserSampleRegressionTests(SimpleTestCase):
    def test_parses_all_production_samples(self):
        if not os.path.isdir(SAMPLES_DIR):
            self.skipTest("samples dir missing")
        samples = [
            f for f in os.listdir(SAMPLES_DIR)
            if f.endswith(".json") and not f.startswith(("t_", "test"))
        ]
        self.assertGreater(len(samples), 0)
        for fname in samples:
            with self.subTest(sample=fname):
                with open(os.path.join(SAMPLES_DIR, fname), encoding="utf-8") as f:
                    payload = json.load(f)
                urunler, container = parse_optimization_payload(payload)
                self.assertGreater(len(urunler), 0)
                for required in ("urun_kodu", "boy", "en", "yukseklik", "agirlik"):
                    self.assertIn(required, urunler[0])

    def test_real_world_0114_yields_known_count(self):
        path = os.path.join(SAMPLES_DIR, "0114.json")
        if not os.path.exists(path):
            self.skipTest("0114.json missing")
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        urunler, _ = parse_optimization_payload(payload)
        self.assertEqual(len(urunler), 381)
