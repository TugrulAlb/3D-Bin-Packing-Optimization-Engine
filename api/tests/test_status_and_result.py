"""GET /status/ + GET /result/ state machine ve serialization."""

import json

from palet_app.models import Optimization, Palet, Urun

from ._helpers import ApiTestBase, make_optimization


class StatusTests(ApiTestBase):
    def test_status_running(self):
        opt = make_optimization(
            islem_durumu=json.dumps({
                "current_step": 2, "total_steps": 5,
                "phase": "mix", "messages": [],
            })
        )
        r = self.client.get(f"/api/v1/optimize/{opt.id}/status/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.data["status"], "running")
        self.assertFalse(r.data["completed"])

    def test_status_completed_includes_summary(self):
        opt = make_optimization(
            tamamlandi=True, toplam_palet=3, single_palet=1, mix_palet=2,
        )
        r = self.client.get(f"/api/v1/optimize/{opt.id}/status/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.data["status"], "completed")
        self.assertEqual(r.data["percent"], 100)
        self.assertIn("summary", r.data)
        self.assertEqual(r.data["summary"]["toplam_palet"], 3)

    def test_status_failed_when_current_step_minus_one(self):
        opt = make_optimization(
            islem_durumu=json.dumps({
                "current_step": -1, "total_steps": 5, "messages": ["err"],
            })
        )
        r = self.client.get(f"/api/v1/optimize/{opt.id}/status/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.data["status"], "failed")

    def test_status_cancelled_when_durum_flag_set(self):
        opt = make_optimization(
            islem_durumu=json.dumps({
                "current_step": 2, "total_steps": 5,
                "phase": "mix", "messages": [], "cancelled": True,
            })
        )
        r = self.client.get(f"/api/v1/optimize/{opt.id}/status/")
        self.assertEqual(r.data["status"], "cancelled")

    def test_status_unknown_id_returns_404(self):
        r = self.client.get("/api/v1/optimize/99999/status/")
        self.assertEqual(r.status_code, 404)


class ResultTests(ApiTestBase):
    def test_not_ready_returns_409(self):
        opt = make_optimization()
        r = self.client.get(f"/api/v1/optimize/{opt.id}/result/")
        self.assertEqual(r.status_code, 409)
        self.assertEqual(r.data["error"]["code"], "NOT_READY")

    def test_serializes_single_palet(self):
        opt = make_optimization(tamamlandi=True, toplam_palet=1, single_palet=1)
        urun = Urun.objects.create(
            urun_kodu="A1", urun_adi="A1",
            boy=40, en=30, yukseklik=25, agirlik=10, mukavemet=200,
        )
        Palet.objects.create(
            optimization=opt, palet_id=1, palet_turu="single",
            custom_en=100, custom_boy=120, custom_max_yukseklik=180,
            custom_max_agirlik=1250,
            toplam_agirlik=10, kullanilan_hacim=30000,
            urun_konumlari={str(urun.id): [0, 0, 0]},
            urun_boyutlari={str(urun.id): [40, 30, 25]},
        )
        r = self.client.get(f"/api/v1/optimize/{opt.id}/result/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.data["paletler"]), 1)
        palet = r.data["paletler"][0]
        self.assertEqual(palet["palet_id"], 1)
        self.assertEqual(palet["palet_turu"], "single")
        self.assertEqual(len(palet["urunler"]), 1)
        u = palet["urunler"][0]
        self.assertEqual(u["urun_kodu"], "A1")
        self.assertEqual(u["position"], {"x": 0.0, "y": 0.0, "z": 0.0})
        self.assertEqual(u["dimensions"], {"boy": 40.0, "en": 30.0, "yukseklik": 25.0})

    def test_serializes_mix_palet_multiple_items(self):
        opt = make_optimization(tamamlandi=True, toplam_palet=1, mix_palet=1)
        u1 = Urun.objects.create(
            urun_kodu="A", urun_adi="A",
            boy=40, en=30, yukseklik=25, agirlik=10, mukavemet=200,
        )
        u2 = Urun.objects.create(
            urun_kodu="B", urun_adi="B",
            boy=20, en=20, yukseklik=20, agirlik=5, mukavemet=200,
        )
        Palet.objects.create(
            optimization=opt, palet_id=1, palet_turu="mix",
            custom_en=100, custom_boy=120, custom_max_yukseklik=180,
            custom_max_agirlik=1250,
            toplam_agirlik=15, kullanilan_hacim=38000,
            urun_konumlari={str(u1.id): [0, 0, 0], str(u2.id): [40, 0, 0]},
            urun_boyutlari={str(u1.id): [40, 30, 25], str(u2.id): [20, 20, 20]},
        )
        r = self.client.get(f"/api/v1/optimize/{opt.id}/result/")
        self.assertEqual(r.status_code, 200)
        palet = r.data["paletler"][0]
        self.assertEqual(palet["palet_turu"], "mix")
        self.assertEqual(palet["urun_sayisi"], 2)
        kods = {u["urun_kodu"] for u in palet["urunler"]}
        self.assertEqual(kods, {"A", "B"})

    def test_result_summary_oran_calculation(self):
        opt = make_optimization(
            tamamlandi=True, toplam_palet=4, single_palet=3, mix_palet=1,
        )
        r = self.client.get(f"/api/v1/optimize/{opt.id}/result/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.data["summary"]["single_palet_oran"], 75.0)
        self.assertEqual(r.data["summary"]["mix_palet_oran"], 25.0)

    def test_result_unknown_id_returns_404(self):
        r = self.client.get("/api/v1/optimize/99999/result/")
        self.assertEqual(r.status_code, 404)
