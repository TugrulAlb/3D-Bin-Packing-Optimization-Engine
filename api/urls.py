from django.urls import path

from . import views


app_name = "api"


urlpatterns = [
    path("v1/health/", views.HealthView.as_view(), name="health"),
    path("v1/optimize/", views.OptimizeCreateView.as_view(), name="optimize-create"),
    path("v1/optimize/<int:job_id>/status/", views.OptimizeStatusView.as_view(), name="optimize-status"),
    path("v1/optimize/<int:job_id>/result/", views.OptimizeResultView.as_view(), name="optimize-result"),
    path("v1/optimize/<int:job_id>/cancel/", views.OptimizeCancelView.as_view(), name="optimize-cancel"),
]
