"""X-API-Key tabanlı DRF authentication."""

import hmac

from django.conf import settings
from rest_framework import authentication, exceptions


class _ApiClient:
    """request.user yerine geçen hafif nesne."""

    is_authenticated = True
    is_anonymous = False
    is_staff = False
    is_superuser = False

    def __init__(self, label):
        self.label = label
        self.username = label
        self.pk = label

    def __str__(self):
        return f"api:{self.label}"


class ApiKeyAuthentication(authentication.BaseAuthentication):
    keyword = "X-API-Key"

    def authenticate(self, request):
        provided = request.META.get("HTTP_X_API_KEY")
        if not provided:
            return None

        keys = getattr(settings, "API_KEYS", {}) or {}
        for label, expected in keys.items():
            if hmac.compare_digest(str(expected), str(provided)):
                return (_ApiClient(label), provided)

        raise exceptions.AuthenticationFailed("Invalid API key")

    def authenticate_header(self, request):
        return self.keyword
