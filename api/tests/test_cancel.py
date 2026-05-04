"""POST /cancel/ akışı."""

import json
from unittest import mock

from ._helpers import ApiTestBase, make_optimization


class CancelEndpointTests(ApiTestBase):
    @mock.patch("api.services.cancel_opt")
    def test_cancel_running_signals_worker(self, mock_cancel):
        opt = make_optimization(
            islem_durumu=json.dumps({
                "current_step": 1, "total_steps": 5,
                "phase": "mix", "messages": [],
            })
        )
        r = self.client.post(f"/api/v1/optimize/{opt.id}/cancel/")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.data["cancelled"])
        mock_cancel.assert_called_once_with(opt.id)

    def test_cancel_completed_returns_409(self):
        opt = make_optimization(tamamlandi=True)
        r = self.client.post(f"/api/v1/optimize/{opt.id}/cancel/")
        self.assertEqual(r.status_code, 409)
        self.assertEqual(r.data["error"]["code"], "ALREADY_TERMINAL")

    def test_cancel_idempotency_second_call_409(self):
        opt = make_optimization(tamamlandi=True)
        first = self.client.post(f"/api/v1/optimize/{opt.id}/cancel/")
        second = self.client.post(f"/api/v1/optimize/{opt.id}/cancel/")
        self.assertEqual(first.status_code, 409)
        self.assertEqual(second.status_code, 409)

    def test_cancel_unknown_id_returns_404(self):
        r = self.client.post("/api/v1/optimize/99999/cancel/")
        self.assertEqual(r.status_code, 404)

    def test_cancel_via_get_also_works(self):
        opt = make_optimization(tamamlandi=True)
        r = self.client.get(f"/api/v1/optimize/{opt.id}/cancel/")
        self.assertEqual(r.status_code, 409)
