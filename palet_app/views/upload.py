"""Yükleme, ürün listesi, ana sayfa view'ları."""

import json
import logging
import os
import tempfile

from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.urls import reverse

from ..services import parse_optimization_payload


logger = logging.getLogger(__name__)


def home_view(request):
    return render(request, 'palet_app/home.html')


def upload_result(request):
    """AJAX ile yüklenen JSON dosyasını işler."""
    if request.method != 'POST' or 'file' not in request.FILES:
        return JsonResponse({'success': False, 'error': 'Dosya yüklenemedi.'}, status=400)

    uploaded_file = request.FILES['file']

    if not uploaded_file.name.lower().endswith('.json'):
        return JsonResponse({'success': False, 'error': 'Yalnızca JSON dosyaları kabul edilir.'}, status=400)

    temp_file_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)

    with open(temp_file_path, 'wb+') as destination:
        for chunk in uploaded_file.chunks():
            destination.write(chunk)

    try:
        with open(temp_file_path, 'r', encoding='utf-8') as f:
            yuklenen_veri = json.load(f)
    except json.JSONDecodeError:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return JsonResponse({'success': False, 'error': 'Geçersiz JSON formatı.'}, status=400)
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    try:
        urun_verileri, container_info = parse_optimization_payload(yuklenen_veri)
    except ValueError as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)
    except Exception:
        logger.exception("Upload parse hatası")
        return JsonResponse({'success': False, 'error': 'Dosya işlenemedi.'}, status=400)

    if container_info:
        request.session['container_info'] = container_info
    request.session['urun_verileri'] = urun_verileri

    return JsonResponse({
        'success': True,
        'message': f'Toplam {len(urun_verileri)} ürün yüklendi.',
        'next_url': reverse('palet_app:urun_listesi'),
    })


def urun_listesi(request):
    """Yüklenen ürünleri listeler."""
    if 'urun_verileri' not in request.session:
        return redirect('palet_app:home')

    urun_verileri = request.session.get('urun_verileri', [])
    container_info = request.session.get('container_info', {})

    urun_gruplari = {}
    for urun in urun_verileri:
        kod = urun['urun_kodu']
        if kod not in urun_gruplari:
            urun_gruplari[kod] = {
                'urun_kodu': kod,
                'urun_adi': urun['urun_adi'],
                'boy': urun['boy'],
                'en': urun['en'],
                'yukseklik': urun['yukseklik'],
                'agirlik': urun['agirlik'],
                'mukavemet': urun.get('mukavemet', 'N/A'),
                'adet': 0,
                'toplam_agirlik': 0,
                'toplam_hacim': 0,
            }
        urun_gruplari[kod]['adet'] += 1
        urun_gruplari[kod]['toplam_agirlik'] += urun['agirlik']
        urun_gruplari[kod]['toplam_hacim'] += (urun['boy'] * urun['en'] * urun['yukseklik'])

    urun_listesi_sorted = sorted(urun_gruplari.values(), key=lambda x: x['urun_kodu'])

    context = {
        'urun_listesi': urun_listesi_sorted,
        'toplam_urun_cesidi': len(urun_listesi_sorted),
        'toplam_paket': len(urun_verileri),
        'container_info': container_info,
    }

    return render(request, 'palet_app/urun_listesi.html', context)
