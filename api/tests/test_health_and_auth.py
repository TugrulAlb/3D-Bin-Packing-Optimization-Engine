"""Health probe ve API key auth davranışı."""

from django.test import TestCase, override_settings
from rest_framework.test import APIClient


class HealthTests(TestCase):
    def test_health_no_auth_required(self):
        c = APIClient()
        r = c.get("/api/v1/health/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.data["status"], "ok")
        self.assertIn("version", r.data)
        self.assertIn("time", r.data)


class AuthTests(TestCase):
    def test_missing_key_returns_401(self):
        c = APIClient()
        r = c.post("/api/v1/optimize/", data={}, format="json")
        self.assertEqual(r.status_code, 401)

    @override_settings(API_KEYS={"test": "real-key"})
    def test_invalid_key_returns_401(self):
        c = APIClient()
        c.credentials(HTTP_X_API_KEY="wrong")
        r = c.post("/api/v1/optimize/", data={}, format="json")
        self.assertEqual(r.status_code, 401)
        self.assertIn("error", r.data)

    @override_settings(API_KEYS={})
    def test_no_keys_configured_blocks_all_requests(self):
        c = APIClient()
        c.credentials(HTTP_X_API_KEY="anything")
        r = c.post("/api/v1/optimize/", data={}, format="json")
        self.assertEqual(r.status_code, 401)
