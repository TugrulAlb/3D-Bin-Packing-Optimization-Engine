"""Standart API hata zarfı.

Tüm DRF exception'lar `{"error": {"code", "message", "details?"}}` formatında döner.
"""

from rest_framework.response import Response
from rest_framework.views import exception_handler


def api_exception_handler(exc, context):
    response = exception_handler(exc, context)
    if response is None:
        return Response(
            {"error": {"code": "INTERNAL_ERROR", "message": str(exc) or exc.__class__.__name__}},
            status=500,
        )

    detail = response.data
    code = getattr(exc, "default_code", "ERROR")
    if isinstance(detail, dict) and "detail" in detail:
        message = str(detail.get("detail", ""))
        details = None
    elif isinstance(detail, dict):
        message = "Validation failed"
        details = detail
    elif isinstance(detail, list):
        message = "; ".join(str(x) for x in detail)
        details = detail
    else:
        message = str(detail)
        details = None

    payload = {"error": {"code": str(code).upper(), "message": message}}
    if details is not None:
        payload["error"]["details"] = details
    response.data = payload
    return response
