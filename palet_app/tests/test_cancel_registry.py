"""cancel_registry — modül-seviyesi state, izolasyon kritik."""

from django.test import SimpleTestCase

from palet_app.workers import cancel_registry
from palet_app.workers.cancel_registry import (
    OptimizationCancelled,
    _reset_for_tests,
    cancel_group,
    cancel_opt,
    check_cancel,
    is_cancelled,
)


class CancelRegistryTests(SimpleTestCase):
    def setUp(self):
        _reset_for_tests()

    def tearDown(self):
        _reset_for_tests()

    def test_cancel_opt_marks_id(self):
        cancel_opt(123)
        self.assertTrue(is_cancelled(opt_id=123))

    def test_unrelated_id_not_cancelled(self):
        cancel_opt(123)
        self.assertFalse(is_cancelled(opt_id=999))

    def test_cancel_group_marks_group(self):
        cancel_group("g1")
        self.assertTrue(is_cancelled(group_id="g1"))

    def test_cancel_group_falsy_input_is_noop(self):
        cancel_group(None)
        cancel_group("")
        self.assertFalse(is_cancelled(group_id=""))

    def test_cancel_opt_idempotent(self):
        cancel_opt(7)
        cancel_opt(7)
        self.assertTrue(is_cancelled(opt_id=7))

    def test_check_cancel_raises(self):
        cancel_opt(99)
        with self.assertRaises(OptimizationCancelled):
            check_cancel(99)

    def test_check_cancel_silent_when_not_marked(self):
        check_cancel(456)

    def test_cancel_opt_coerces_string_id(self):
        cancel_opt("42")
        self.assertTrue(is_cancelled(opt_id=42))
        self.assertTrue(is_cancelled(opt_id="42"))

    def test_reset_for_tests_clears_state(self):
        cancel_opt(1)
        cancel_group("g")
        _reset_for_tests()
        self.assertFalse(is_cancelled(opt_id=1))
        self.assertFalse(is_cancelled(group_id="g"))
