"""api.services derive_status state machine + cancel_job + elapsed_seconds."""

import json
from unittest import mock

from django.test import TestCase
from django.utils import timezone

from api.services import cancel_job, derive_status, elapsed_seconds
from palet_app.workers.cancel_registry import _reset_for_tests as cancel_reset

from ._helpers import make_optimization


class DeriveStatusTests(TestCase):
    def setUp(self):
        cancel_reset()

    def tearDown(self):
        cancel_reset()

    def test_completed_state(self):
        opt = make_optimization(tamamlandi=True)
        s = derive_status(opt)
        self.assertEqual(s["status"], "completed")
        self.assertTrue(s["completed"])
        self.assertEqual(s["percent"], 100)

    def test_cancelled_overrides_running(self):
        opt = make_optimization(
            islem_durumu=json.dumps({
                "current_step": 2, "total_steps": 5,
                "phase": "mix", "messages": [], "cancelled": True,
            })
        )
        s = derive_status(opt)
        self.assertEqual(s["status"], "cancelled")
        self.assertTrue(s["cancelled"])

    def test_failed_when_current_step_minus_one(self):
        opt = make_optimization(
            islem_durumu=json.dumps({
                "current_step": -1, "total_steps": 5, "messages": [],
            })
        )
        s = derive_status(opt)
        self.assertEqual(s["status"], "failed")

    def test_running_when_phase_set(self):
        opt = make_optimization(
            islem_durumu=json.dumps({
                "current_step": 1, "total_steps": 5,
                "phase": "single", "messages": [],
            })
        )
        s = derive_status(opt)
        self.assertEqual(s["status"], "running")

    def test_queued_when_no_phase(self):
        opt = make_optimization()
        s = derive_status(opt)
        self.assertEqual(s["status"], "queued")

    def test_completed_takes_priority_over_cancelled_flag(self):
        opt = make_optimization(
            tamamlandi=True,
            islem_durumu=json.dumps({
                "current_step": 5, "total_steps": 5,
                "messages": [], "cancelled": True,
            }),
        )
        s = derive_status(opt)
        self.assertEqual(s["status"], "completed")


class CancelJobTests(TestCase):
    def setUp(self):
        cancel_reset()

    def tearDown(self):
        cancel_reset()

    @mock.patch("api.services.cancel_opt")
    def test_running_job_signals_cancel(self, mock_cancel):
        opt = make_optimization(
            islem_durumu=json.dumps({
                "current_step": 1, "total_steps": 5, "phase": "mix", "messages": [],
            })
        )
        result = cancel_job(opt)
        self.assertFalse(result["already_terminal"])
        self.assertEqual(result["previous_status"], "running")
        mock_cancel.assert_called_once_with(opt.id)

    @mock.patch("api.services.cancel_opt")
    def test_completed_job_returns_already_terminal(self, mock_cancel):
        opt = make_optimization(tamamlandi=True)
        result = cancel_job(opt)
        self.assertTrue(result["already_terminal"])
        self.assertEqual(result["previous_status"], "completed")
        mock_cancel.assert_not_called()


class ElapsedSecondsTests(TestCase):
    def test_uses_now_when_bitis_none(self):
        opt = make_optimization()
        elapsed = elapsed_seconds(opt)
        self.assertIsNotNone(elapsed)
        self.assertGreaterEqual(elapsed, 0.0)

    def test_uses_bitis_when_set(self):
        now = timezone.now()
        opt = make_optimization(tamamlandi=True)
        opt.bitis_zamani = now
        opt.save()
        elapsed = elapsed_seconds(opt)
        self.assertIsNotNone(elapsed)
        self.assertGreaterEqual(elapsed, 0.0)

    def test_returns_none_without_baslangic(self):
        opt = make_optimization()
        opt.baslangic_zamani = None
        # NOT NULL DB constraint var; in-memory kontrat test'i.
        self.assertIsNone(elapsed_seconds(opt))
