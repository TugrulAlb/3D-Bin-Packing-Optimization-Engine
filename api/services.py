"""API servis katmanı — orkestrasyon (DB + worker thread)."""

import json
import logging
from threading import Thread

from django.conf import settings
from django.db import transaction
from django.utils import timezone

from palet_app.models import Optimization, PHASE_RANGES
from palet_app.services import parse_optimization_payload
from palet_app.workers import (
    cancel_opt,
    is_cancelled,
    normalize_progress,
    run_optimization,
)


logger = logging.getLogger("api.services")


class CapacityError(Exception):
    """Eşzamanlı iş limiti aşıldığında fırlatılır."""


def _running_jobs_count() -> int:
    return Optimization.objects.filter(tamamlandi=False).count()


def start_optimization(payload: dict, *, api_client_label: str = "api") -> Optimization:
    """JSON payload'tan iş başlatır, Optimization satırı döner.

    `payload` `OptimizeRequestSerializer.validated_data` formatındadır.
    Geçersiz format -> ValueError ; eşzamanlı limit -> CapacityError.
    """
    max_concurrent = getattr(settings, "API_MAX_CONCURRENT_JOBS", 4)
    if max_concurrent and _running_jobs_count() >= max_concurrent:
        raise CapacityError(f"Eşzamanlı iş limiti dolu ({max_concurrent}).")

    urun_verileri, container_info = parse_optimization_payload(payload)

    container = payload.get("container") or {}
    container_length = container_info.get("length", container.get("length", 120))
    container_width = container_info.get("width", container.get("width", 100))
    container_height = container_info.get("height", container.get("height", 180))
    container_weight = container_info.get("weight", container.get("weight", 1250))

    algorithm = (payload.get("algorithm") or payload.get("algoritma") or "greedy").strip().lower()
    ga_mode = (payload.get("ga_mode") or "balanced").strip().lower()
    if algorithm not in ("greedy", "genetic", "differential_evolution"):
        algorithm = "greedy"
    if ga_mode not in ("fast", "balanced", "quality"):
        ga_mode = "balanced"

    with transaction.atomic():
        opt = Optimization.objects.create(
            palet_tipi=None,
            container_length=container_length,
            container_width=container_width,
            container_height=container_height,
            container_weight=container_weight,
            algoritma=algorithm,
            islem_durumu=json.dumps({
                "current_step": 0,
                "total_steps": 5,
                "messages": [f"[API:{api_client_label}] İş kuyruğa alındı"],
                "api_client": api_client_label,
                "ga_mode": ga_mode,
                "product_count": len(urun_verileri),
            }),
        )

    container_dict = {
        "length": container_length,
        "width": container_width,
        "height": container_height,
        "weight": container_weight,
    }

    try:
        thread = Thread(
            target=run_optimization,
            args=(urun_verileri, container_dict, opt.id, algorithm, ga_mode),
            daemon=True,
        )
        thread.start()
    except Exception as e:
        logger.exception("API: thread baslatilamadi (opt %s): %s", opt.id, e)
        raise

    logger.info(
        "api.optimize.submitted",
        extra={
            "job_id": opt.id, "client": api_client_label, "algorithm": algorithm,
            "ga_mode": ga_mode, "product_count": len(urun_verileri),
        },
    )
    return opt


def derive_status(opt: Optimization) -> dict:
    """Optimization satırından stabil status sözlüğü üretir."""
    durum = opt.get_islem_durumu()
    cancelled = bool(durum.get("cancelled")) or is_cancelled(opt_id=opt.id)
    error_step = durum.get("current_step", 0) == -1

    if opt.tamamlandi:
        status = "completed"
    elif cancelled:
        status = "cancelled"
    elif error_step:
        status = "failed"
    elif durum.get("phase"):
        status = "running"
    else:
        status = "queued"

    cur, tot, pct = normalize_progress(
        durum.get("current_step", 0),
        durum.get("total_steps", 5),
        completed=opt.tamamlandi,
        durum=durum,
        error=(status == "failed"),
    )
    phase = durum.get("phase") or ""
    phase_label = "Tamamlandı" if opt.tamamlandi else PHASE_RANGES.get(phase, (0, 0, "Hazırlanıyor"))[2]

    return {
        "status": status,
        "completed": opt.tamamlandi,
        "cancelled": cancelled,
        "phase": phase or None,
        "phase_label": phase_label,
        "percent": 100 if opt.tamamlandi else pct,
        "current_step": cur,
        "total_steps": tot,
        "messages": durum.get("messages", []) or [],
        "raw": durum,
    }


def cancel_job(opt: Optimization) -> dict:
    """İptal sinyali yollar; daha önce bitmişse `already_terminal=True` döner."""
    state = derive_status(opt)
    prev = state["status"]
    if prev in ("completed", "failed", "cancelled"):
        return {"already_terminal": True, "previous_status": prev}
    cancel_opt(opt.id)
    logger.info("api.optimize.cancel_signaled", extra={"job_id": opt.id})
    return {"already_terminal": False, "previous_status": prev}


def elapsed_seconds(opt: Optimization):
    if not opt.baslangic_zamani:
        return None
    end = opt.bitis_zamani or timezone.now()
    return round((end - opt.baslangic_zamani).total_seconds(), 2)


def iso(dt):
    return dt.isoformat() if dt else None
