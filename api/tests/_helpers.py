"""Test helper'ları — fixture loader, opt factory, auth client."""

import json
import os

from django.test import override_settings
from django.test.testcases import TestCase
from rest_framework.test import APIClient

from palet_app.models import Optimization
from palet_app.workers.cancel_registry import _reset_for_tests as cancel_reset


SAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "samples",
)


def load_sample(name="0114.json"):
    with open(os.path.join(SAMPLES_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def make_optimization(**overrides):
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


@override_settings(API_KEYS={"test": "secret-test-key"}, API_MAX_CONCURRENT_JOBS=10)
class ApiTestBase(TestCase):
    """Auth header + cancel registry reset baseline."""

    def setUp(self):
        cancel_reset()
        self.client = APIClient()
        self.client.credentials(HTTP_X_API_KEY="secret-test-key")

    def tearDown(self):
        cancel_reset()
