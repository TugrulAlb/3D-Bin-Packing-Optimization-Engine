"""REST API view'ları (DRF APIView)."""

import logging

from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework import status as http
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from palet_app.models import Optimization, Palet

from .serializers import OptimizeRequestSerializer, serialize_paletler
from .services import (
    CapacityError,
    cancel_job,
    derive_status,
    elapsed_seconds,
    iso,
    start_optimization,
)


logger = logging.getLogger("api.views")


def _client_label(request) -> str:
    user = getattr(request, "user", None)
    return getattr(user, "label", None) or "anonymous"


class HealthView(APIView):
    """Liveness probe — auth gerektirmez."""

    authentication_classes = []
    permission_classes = [AllowAny]

    def get(self, request):
        return Response({
            "status": "ok",
            "service": "dayframe-bin-packing",
            "version": "1.0.0",
            "time": timezone.now().isoformat(),
        })


class OptimizeCreateView(APIView):
    throttle_scope = "submit"

    def post(self, request):
        ser = OptimizeRequestSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        try:
            opt = start_optimization(
                ser.validated_data,
                api_client_label=_client_label(request),
            )
        except ValueError as e:
            return Response(
                {"error": {"code": "VALIDATION_ERROR", "message": str(e)}},
                status=http.HTTP_400_BAD_REQUEST,
            )
        except CapacityError as e:
            return Response(
                {"error": {"code": "CAPACITY_EXCEEDED", "message": str(e)}},
                status=http.HTTP_429_TOO_MANY_REQUESTS,
                headers={"Retry-After": "30"},
            )

        body = {
            "job_id": opt.id,
            "status": "queued",
            "algorithm": opt.algoritma,
            "ga_mode": ser.validated_data.get("ga_mode", "balanced"),
            "product_count": len(ser.validated_data.get("details", [])),
            "container": {
                "length": opt.container_length,
                "width": opt.container_width,
                "height": opt.container_height,
                "weight": opt.container_weight,
            },
            "links": {
                "status": f"/api/v1/optimize/{opt.id}/status/",
                "result": f"/api/v1/optimize/{opt.id}/result/",
                "cancel": f"/api/v1/optimize/{opt.id}/cancel/",
            },
            "submitted_at": iso(opt.baslangic_zamani),
        }
        return Response(body, status=http.HTTP_202_ACCEPTED)


class OptimizeStatusView(APIView):
    throttle_scope = "status"

    def get(self, request, job_id):
        opt = get_object_or_404(Optimization, id=job_id)
        s = derive_status(opt)

        body = {
            "job_id": opt.id,
            "status": s["status"],
            "completed": s["completed"],
            "cancelled": s["cancelled"],
            "phase": s["phase"],
            "phase_label": s["phase_label"],
            "percent": s["percent"],
            "current_step": s["current_step"],
            "total_steps": s["total_steps"],
            "messages": s["messages"][-50:],
            "elapsed_sec": elapsed_seconds(opt),
            "started_at": iso(opt.baslangic_zamani),
            "finished_at": iso(opt.bitis_zamani),
        }
        if opt.tamamlandi:
            body["summary"] = {
                "toplam_palet": opt.toplam_palet,
                "single_palet": opt.single_palet,
                "mix_palet": opt.mix_palet,
                "yerlesmemis_urun_sayisi": len(opt.yerlesmemis_urunler or []),
            }
            body["result_url"] = f"/api/v1/optimize/{opt.id}/result/"
        return Response(body)


class OptimizeResultView(APIView):
    throttle_scope = "result"

    def get(self, request, job_id):
        opt = get_object_or_404(Optimization, id=job_id)
        s = derive_status(opt)
        if s["status"] != "completed":
            return Response(
                {"error": {
                    "code": "NOT_READY",
                    "message": "İş henüz tamamlanmadı.",
                    "status": s["status"],
                    "percent": s["percent"],
                }},
                status=http.HTTP_409_CONFLICT,
            )

        paletler = (Palet.objects
                    .filter(optimization=opt)
                    .order_by("palet_id"))

        elapsed = elapsed_seconds(opt)
        toplam = opt.toplam_palet or 0
        single_oran = round(opt.single_palet / toplam * 100, 2) if toplam else 0.0
        mix_oran = round(opt.mix_palet / toplam * 100, 2) if toplam else 0.0

        return Response({
            "job_id": opt.id,
            "status": "completed",
            "algorithm": opt.algoritma,
            "container": {
                "length": opt.container_length,
                "width": opt.container_width,
                "height": opt.container_height,
                "weight": opt.container_weight,
            },
            "summary": {
                "toplam_palet": opt.toplam_palet,
                "single_palet": opt.single_palet,
                "mix_palet": opt.mix_palet,
                "single_palet_oran": single_oran,
                "mix_palet_oran": mix_oran,
                "yerlesmemis_urun_sayisi": len(opt.yerlesmemis_urunler or []),
                "elapsed_sec": elapsed,
                "started_at": iso(opt.baslangic_zamani),
                "finished_at": iso(opt.bitis_zamani),
            },
            "paletler": serialize_paletler(paletler),
            "yerlesmemis_urunler": opt.yerlesmemis_urunler or [],
        })


class OptimizeCancelView(APIView):
    throttle_scope = "cancel"

    def post(self, request, job_id):
        return self._do_cancel(job_id)

    def get(self, request, job_id):
        return self._do_cancel(job_id)

    def _do_cancel(self, job_id):
        opt = get_object_or_404(Optimization, id=job_id)
        result = cancel_job(opt)
        if result["already_terminal"]:
            return Response(
                {"error": {
                    "code": "ALREADY_TERMINAL",
                    "message": f"İş zaten {result['previous_status']} durumunda.",
                    "status": result["previous_status"],
                }},
                status=http.HTTP_409_CONFLICT,
            )
        return Response({
            "job_id": opt.id,
            "cancelled": True,
            "previous_status": result["previous_status"],
            "note": "İptal sinyali gönderildi; worker bir sonraki faz sınırında durur.",
        })
