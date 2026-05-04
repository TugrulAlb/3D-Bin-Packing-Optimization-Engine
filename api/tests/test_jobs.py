"""POST /optimize/ + concurrency cap."""

from unittest import mock

from django.test import override_settings

from palet_app.models import Optimization

from ._helpers import ApiTestBase


def _payload():
    return {
        "container": {"length": 120, "width": 100, "height": 180, "weight": 1250},
        "details": [{
            "product": {
                "code": "TEST",
                "package_length": 40, "package_width": 30,
                "package_height": 25, "package_weight": 10,
            },
            "package_quantity": 3,
        }],
        "algorithm": "greedy",
    }


class CreateJobTests(ApiTestBase):
    @mock.patch("api.services.Thread")
    def test_returns_202_and_creates_optimization_row(self, mock_thread):
        mock_thread.return_value.start = mock.Mock()
        r = self.client.post("/api/v1/optimize/", data=_payload(), format="json")
        self.assertEqual(r.status_code, 202)
        self.assertIn("job_id", r.data)
        self.assertEqual(r.data["status"], "queued")
        self.assertEqual(Optimization.objects.count(), 1)
        opt = Optimization.objects.first()
        self.assertEqual(opt.algoritma, "greedy")
        mock_thread.assert_called_once()
        mock_thread.return_value.start.assert_called_once()

    @mock.patch("api.services.Thread")
    def test_response_links_match_optimization_id(self, mock_thread):
        mock_thread.return_value.start = mock.Mock()
        r = self.client.post("/api/v1/optimize/", data=_payload(), format="json")
        opt_id = r.data["job_id"]
        self.assertEqual(r.data["links"]["status"], f"/api/v1/optimize/{opt_id}/status/")
        self.assertEqual(r.data["links"]["result"], f"/api/v1/optimize/{opt_id}/result/")
        self.assertEqual(r.data["links"]["cancel"], f"/api/v1/optimize/{opt_id}/cancel/")


class ConcurrencyLimitTests(ApiTestBase):
    @override_settings(API_KEYS={"test": "secret-test-key"}, API_MAX_CONCURRENT_JOBS=1)
    def test_capacity_exceeded_returns_429(self):
        Optimization.objects.create(
            container_length=120, container_width=100,
            container_height=180, container_weight=1250,
            algoritma="greedy", tamamlandi=False,
        )
        r = self.client.post("/api/v1/optimize/", data=_payload(), format="json")
        self.assertEqual(r.status_code, 429)
        self.assertEqual(r.data["error"]["code"], "CAPACITY_EXCEEDED")

    @mock.patch("api.services.Thread")
    @override_settings(API_KEYS={"test": "secret-test-key"}, API_MAX_CONCURRENT_JOBS=1)
    def test_capacity_freed_after_completion_accepts_new(self, mock_thread):
        mock_thread.return_value.start = mock.Mock()
        Optimization.objects.create(
            container_length=120, container_width=100,
            container_height=180, container_weight=1250,
            algoritma="greedy", tamamlandi=True,
        )
        r = self.client.post("/api/v1/optimize/", data=_payload(), format="json")
        self.assertEqual(r.status_code, 202)
