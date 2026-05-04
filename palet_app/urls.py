from django.urls import path

from . import views


app_name = 'palet_app'

urlpatterns = [
    path('', views.home_view, name='home'),
    path('urun-listesi/', views.urun_listesi, name='urun_listesi'),
    path('isleniyor/', views.processing, name='processing'),
    path('analiz/', views.analysis, name='analysis'),
    path('palet-detay/<int:palet_id>/', views.palet_detail, name='palet_detail'),
    path('palet-detay/<int:palet_id>/3d-data/', views.palet_3d_data, name='palet_3d_data'),

    path('yukle-sonuc/', views.upload_result, name='upload_result'),
    path('yerlestirme-baslat/', views.start_placement, name='start_placement'),
    path('optimizasyon-durumu/', views.optimization_status, name='optimization_status'),
    path('optimizasyon-iptal/', views.cancel_optimization, name='cancel_optimization'),

    path('benchmark/baslat/', views.start_benchmark, name='start_benchmark'),
    path('benchmark/isleniyor/', views.benchmark_processing, name='benchmark_processing'),
    path('benchmark/durum/', views.benchmark_status, name='benchmark_status'),
    path('benchmark/sonuc/', views.benchmark_result, name='benchmark_result'),
    path('benchmark/sec/<int:optimization_id>/', views.benchmark_select, name='benchmark_select'),
    path('benchmark/iptal/', views.cancel_benchmark, name='cancel_benchmark'),
]
