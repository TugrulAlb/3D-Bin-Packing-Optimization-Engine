"""API entegrasyon testleri."""

import json
import os
from unittest import mock

from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from palet_app.models import Optimization, Palet, Urun


SAMPLE_JSON = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data", "samples", "0114.json",
)


def load_sample():
    with open(SAMPLE_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


@override_settings(API_KEYS={"test": "secret-test-key"}, API_MAX_CONCURRENT_JOBS=10)
class ApiBaseTest(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.client.credentials(HTTP_X_API_KEY="secret-test-key")


class HealthTest(TestCase):
    def test_health_no_auth(self):
        c = APIClient()
        r = c.get("/api/v1/health/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.data["status"], "ok")


class AuthTest(TestCase):
    def test_missing_key(self):
        c = APIClient()
        r = c.post("/api/v1/optimize/", data={}, format="json")
        self.assertEqual(r.status_code, 401)

    @override_settings(API_KEYS={"test": "real-key"})
    def test_invalid_key(self):
        c = APIClient()
        c.credentials(HTTP_X_API_KEY="wrong")
        r = c.post("/api/v1/optimize/", data={}, format="json")
        self.assertEqual(r.status_code, 401)
        self.assertIn("error", r.data)


class ValidationTest(ApiBaseTest):
    def test_missing_container(self):
        r = self.client.post("/api/v1/optimize/", data={"details": []}, format="json")
        self.assertEqual(r.status_code, 400)

    def test_empty_details(self):
        r = self.client.post("/api/v1/optimize/", data={
            "container": {"length": 120, "width": 100, "height": 180, "weight": 1250},
            "details": [],
        }, format="json")
        self.assertEqual(r.status_code, 400)

    def test_unknown_algorithm(self):
        r = self.client.post("/api/v1/optimize/", data={
            "container": {"length": 120, "width": 100, "height": 180, "weight": 1250},
            "details": [{"product": {"code": "X", "package_length": 10, "package_width": 10,
                                     "package_height": 10, "package_weight": 1},
                          "package_quantity": 1}],
            "algorithm": "quantum",
        }, format="json")
        self.assertEqual(r.status_code, 400)

    def test_oversize_item(self):
        r = self.client.post("/api/v1/optimize/", data={
            "container": {"length": 120, "width": 100, "height": 180, "weight": 1250},
            "details": [{"product": {"code": "X", "package_length": 200, "package_width": 50,
                                     "package_height": 50, "package_weight": 1},
                          "package_quantity": 1}],
        }, format="json")
        self.assertEqual(r.status_code, 400)


class CreateJobTest(ApiBaseTest):
    @mock.patch("api.services.Thread")
    def test_creates_optimization_and_returns_202(self, mock_thread):
        mock_thread.return_value.start = mock.Mock()
        payload = {
            "container": {"length": 120, "width": 100, "height": 180, "weight": 1250},
            "details": [
                {"product": {"code": "TEST",
                             "package_length": 40, "package_width": 30,
                             "package_height": 25, "package_weight": 10},
                 "package_quantity": 3},
            ],
            "algorithm": "greedy",
        }
        r = self.client.post("/api/v1/optimize/", data=payload, format="json")
        self.assertEqual(r.status_code, 202, r.data)
        self.assertIn("job_id", r.data)
        self.assertEqual(r.data["status"], "queued")
        self.assertEqual(Optimization.objects.count(), 1)
        opt = Optimization.objects.first()
        self.assertEqual(opt.algoritma, "greedy")
        mock_thread.assert_called_once()
        mock_thread.return_value.start.assert_called_once()


class StatusTest(ApiBaseTest):
    def _make_opt(self, **kwargs):
        defaults = dict(
            container_length=120, container_width=100,
            container_height=180, container_weight=1250,
            algoritma="greedy",
            islem_durumu='{"current_step": 0, "total_steps": 5, "messages": []}',
        )
        defaults.update(kwargs)
        return Optimization.objects.create(**defaults)

    def test_running(self):
        opt = self._make_opt(islem_durumu='{"current_step": 2, "total_steps": 5, "phase": "mix", "messages": []}')
        r = self.client.get(f"/api/v1/optimize/{opt.id}/status/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.data["status"], "running")
        self.assertFalse(r.data["completed"])

    def test_completed(self):
        opt = self._make_opt(tamamlandi=True, toplam_palet=3, single_palet=1, mix_palet=2)
        r = self.client.get(f"/api/v1/optimize/{opt.id}/status/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.data["status"], "completed")
        self.assertEqual(r.data["percent"], 100)
        self.assertIn("summary", r.data)

    def test_unknown_id(self):
        r = self.client.get("/api/v1/optimize/99999/status/")
        self.assertEqual(r.status_code, 404)


class ResultTest(ApiBaseTest):
    def test_not_ready(self):
        opt = Optimization.objects.create(
            container_length=120, container_width=100,
            container_height=180, container_weight=1250,
            algoritma="greedy",
        )
        r = self.client.get(f"/api/v1/optimize/{opt.id}/result/")
        self.assertEqual(r.status_code, 409)
        self.assertEqual(r.data["error"]["code"], "NOT_READY")

    def test_serializes_palet(self):
        opt = Optimization.objects.create(
            container_length=120, container_width=100,
            container_height=180, container_weight=1250,
            algoritma="greedy",
            tamamlandi=True,
            toplam_palet=1, single_palet=1, mix_palet=0,
        )
        urun = Urun.objects.create(urun_kodu="A1", urun_adi="A1",
                                   boy=40, en=30, yukseklik=25, agirlik=10,
                                   mukavemet=200,
                                   donus_serbest=True, istiflenebilir=True)
        Palet.objects.create(
            optimization=opt, palet_id=1, palet_turu="single",
            custom_en=100, custom_boy=120, custom_max_yukseklik=180,
            custom_max_agirlik=1250,
            toplam_agirlik=10, kullanilan_hacim=30000,
            urun_konumlari={str(urun.id): [0, 0, 0]},
            urun_boyutlari={str(urun.id): [40, 30, 25]},
        )
        r = self.client.get(f"/api/v1/optimize/{opt.id}/result/")
        self.assertEqual(r.status_code, 200, r.data)
        self.assertEqual(len(r.data["paletler"]), 1)
        palet = r.data["paletler"][0]
        self.assertEqual(palet["palet_id"], 1)
        self.assertEqual(palet["palet_turu"], "single")
        self.assertEqual(len(palet["urunler"]), 1)
        self.assertEqual(palet["urunler"][0]["urun_kodu"], "A1")
        self.assertEqual(palet["urunler"][0]["position"], {"x": 0.0, "y": 0.0, "z": 0.0})


class CancelTest(ApiBaseTest):
    @mock.patch("api.services.cancel_opt")
    def test_cancel_running(self, mock_cancel):
        opt = Optimization.objects.create(
            container_length=120, container_width=100,
            container_height=180, container_weight=1250,
            algoritma="greedy",
            islem_durumu='{"current_step": 1, "total_steps": 5, "phase": "mix", "messages": []}',
        )
        r = self.client.post(f"/api/v1/optimize/{opt.id}/cancel/")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.data["cancelled"])
        mock_cancel.assert_called_once_with(opt.id)

    def test_cancel_already_completed(self):
        opt = Optimization.objects.create(
            container_length=120, container_width=100,
            container_height=180, container_weight=1250,
            algoritma="greedy", tamamlandi=True,
        )
        r = self.client.post(f"/api/v1/optimize/{opt.id}/cancel/")
        self.assertEqual(r.status_code, 409)


class ParserTest(TestCase):
    """Tüm sample dosyaları için parser regression."""

    SAMPLES_DIR = os.path.join(os.path.dirname(SAMPLE_JSON))

    def test_parses_all_samples(self):
        from palet_app.services import parse_optimization_payload
        if not os.path.isdir(self.SAMPLES_DIR):
            self.skipTest("samples dir not found")
        sample_files = [
            f for f in os.listdir(self.SAMPLES_DIR)
            if f.endswith(".json") and not f.startswith(("t_", "test"))
        ]
        self.assertGreater(len(sample_files), 0, "No sample JSONs found")
        for fname in sample_files:
            with self.subTest(sample=fname):
                with open(os.path.join(self.SAMPLES_DIR, fname), encoding="utf-8") as f:
                    payload = json.load(f)
                urun_verileri, container = parse_optimization_payload(payload)
                self.assertGreater(len(urun_verileri), 0, f"{fname}: empty parse")
                for f_ in ("urun_kodu", "boy", "en", "yukseklik", "agirlik"):
                    self.assertIn(f_, urun_verileri[0])

    def test_rejects_invalid_payload(self):
        from palet_app.services import parse_optimization_payload
        with self.assertRaises(ValueError):
            parse_optimization_payload({"foo": "bar"})
        with self.assertRaises(ValueError):
            parse_optimization_payload([])

    def test_rejects_excessive_quantity(self):
        from palet_app.services import parse_optimization_payload
        payload = {
            "container": {"length": 120, "width": 100, "height": 180, "weight": 1250},
            "details": [{
                "product": {"code": "X",
                            "package_length": 10, "package_width": 10,
                            "package_height": 10, "package_weight": 1},
                "package_quantity": 999_999_999,
            }],
        }
        with self.assertRaises(ValueError):
            parse_optimization_payload(payload)


class ConcurrencyLimitTest(ApiBaseTest):
    @override_settings(API_KEYS={"test": "secret-test-key"}, API_MAX_CONCURRENT_JOBS=1)
    def test_capacity_exceeded(self):
        Optimization.objects.create(
            container_length=120, container_width=100,
            container_height=180, container_weight=1250,
            algoritma="greedy", tamamlandi=False,
        )
        payload = {
            "container": {"length": 120, "width": 100, "height": 180, "weight": 1250},
            "details": [{"product": {"code": "X",
                                     "package_length": 10, "package_width": 10,
                                     "package_height": 10, "package_weight": 1},
                          "package_quantity": 1}],
        }
        r = self.client.post("/api/v1/optimize/", data=payload, format="json")
        self.assertEqual(r.status_code, 429)
        self.assertEqual(r.data["error"]["code"], "CAPACITY_EXCEEDED")
