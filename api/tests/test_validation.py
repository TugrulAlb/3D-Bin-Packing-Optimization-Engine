"""POST /api/v1/optimize/ validation katmanı."""

from ._helpers import ApiTestBase


def _detail(**overrides):
    product = {
        "code": "X", "package_length": 40, "package_width": 30,
        "package_height": 25, "package_weight": 10,
    }
    base = {"product": product, "package_quantity": 1}
    base.update(overrides)
    return base


class ValidationTests(ApiTestBase):
    def test_missing_container_returns_400(self):
        r = self.client.post("/api/v1/optimize/", data={"details": []}, format="json")
        self.assertEqual(r.status_code, 400)

    def test_empty_details_returns_400(self):
        r = self.client.post("/api/v1/optimize/", data={
            "container": {"length": 120, "width": 100, "height": 180, "weight": 1250},
            "details": [],
        }, format="json")
        self.assertEqual(r.status_code, 400)

    def test_unknown_algorithm_returns_400(self):
        r = self.client.post("/api/v1/optimize/", data={
            "container": {"length": 120, "width": 100, "height": 180, "weight": 1250},
            "details": [_detail()],
            "algorithm": "quantum",
        }, format="json")
        self.assertEqual(r.status_code, 400)

    def test_oversize_item_returns_400(self):
        r = self.client.post("/api/v1/optimize/", data={
            "container": {"length": 120, "width": 100, "height": 180, "weight": 1250},
            "details": [_detail(product={
                "code": "BIG",
                "package_length": 999, "package_width": 50,
                "package_height": 50, "package_weight": 1,
            })],
        }, format="json")
        self.assertEqual(r.status_code, 400)

    def test_zero_container_dimension_returns_400(self):
        r = self.client.post("/api/v1/optimize/", data={
            "container": {"length": 0, "width": 100, "height": 180, "weight": 1250},
            "details": [_detail()],
        }, format="json")
        self.assertEqual(r.status_code, 400)

    def test_pq_zero_and_qty_zero_returns_400(self):
        r = self.client.post("/api/v1/optimize/", data={
            "container": {"length": 120, "width": 100, "height": 180, "weight": 1250},
            "details": [{
                "product": {
                    "code": "X", "unit_length": 5, "unit_width": 5,
                    "unit_height": 5, "unit_weight": 0.1,
                },
                "package_quantity": 0,
                "quantity": 0,
            }],
        }, format="json")
        self.assertEqual(r.status_code, 400)
