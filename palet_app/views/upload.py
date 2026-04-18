"""Yükleme, ürün listesi, ana sayfa view'ları."""

import json
import os
import tempfile

from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.urls import reverse


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

        os.remove(temp_file_path)

        urun_verileri = []

        if isinstance(yuklenen_veri, dict) and 'details' in yuklenen_veri:
            detaylar = yuklenen_veri.get('details', [])
            container_info = yuklenen_veri.get('container', {})
            try:
                palet_id = yuklenen_veri.get('id')
                if palet_id is not None:
                    container_info['palet_id'] = palet_id
            except Exception:
                pass

            request.session['container_info'] = container_info

            def to_float(x, default=0.0):
                try:
                    return float(x) if x is not None else default
                except (TypeError, ValueError):
                    return default

            for detail in detaylar:
                product = detail.get('product', {})
                package_quantity = detail.get('package_quantity')
                quantity = detail.get('quantity', 0)
                unit_id = detail.get('unit_id', 'ADET')

                code = product.get('code', product.get('id', 'UNKNOWN'))

                package_length = to_float(product.get('package_length'))
                package_width = to_float(product.get('package_width'))
                package_height = to_float(product.get('package_height'))
                package_weight = to_float(product.get('package_weight'))

                unit_length = to_float(product.get('unit_length'))
                unit_width = to_float(product.get('unit_width'))
                unit_height = to_float(product.get('unit_height'))
                unit_weight = to_float(product.get('unit_weight'))

                mukavemet = to_float(product.get('package_max_stack_weight'), default=100000)
                if mukavemet == 0:
                    mukavemet = 100000

                if package_quantity is None or package_quantity <= 0:
                    if unit_id == 'KG' and unit_weight > 0:
                        adet_urun = int(quantity / unit_weight)
                    else:
                        adet_urun = int(quantity)

                    for _ in range(adet_urun):
                        urun_verileri.append({
                            'urun_kodu': str(code),
                            'urun_adi': f"{code}",
                            'boy': unit_length,
                            'en': unit_width,
                            'yukseklik': unit_height,
                            'agirlik': unit_weight,
                            'mukavemet': mukavemet,
                            'donus_serbest': True,
                            'istiflenebilir': True,
                            'package_quantity': None,
                            'quantity': to_float(quantity),
                            'unit_length': unit_length,
                            'unit_width': unit_width,
                            'unit_height': unit_height,
                            'unit_weight': unit_weight,
                        })
                else:
                    for _ in range(package_quantity):
                        urun_verileri.append({
                            'urun_kodu': str(code),
                            'urun_adi': f"{code}",
                            'boy': package_length,
                            'en': package_width,
                            'yukseklik': package_height,
                            'agirlik': package_weight,
                            'mukavemet': mukavemet,
                            'donus_serbest': True,
                            'istiflenebilir': True,
                            'package_quantity': package_quantity,
                            'quantity': to_float(quantity),
                            'unit_length': unit_length,
                            'unit_width': unit_width,
                            'unit_height': unit_height,
                            'unit_weight': unit_weight,
                        })

        elif isinstance(yuklenen_veri, list):
            urun_verileri = yuklenen_veri
        else:
            return JsonResponse({'success': False, 'error': 'Geçersiz JSON formatı. Desteklenen format: {"details": [...]}'}, status=400)

        if not isinstance(urun_verileri, list) or len(urun_verileri) == 0:
            return JsonResponse({'success': False, 'error': 'Geçersiz JSON formatı. Ürün listesi boş veya hatalı.'}, status=400)

        required_fields = ['urun_kodu', 'urun_adi', 'boy', 'en', 'yukseklik', 'agirlik']
        for urun in urun_verileri:
            for field in required_fields:
                if field not in urun:
                    return JsonResponse({'success': False, 'error': f'Eksik alan: {field}'}, status=400)

        request.session['urun_verileri'] = urun_verileri

        return JsonResponse({
            'success': True,
            'message': f'Toplam {len(urun_verileri)} ürün yüklendi.',
            'next_url': reverse('palet_app:urun_listesi'),
        })

    except json.JSONDecodeError:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return JsonResponse({'success': False, 'error': 'Geçersiz JSON formatı.'}, status=400)
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return JsonResponse({'success': False, 'error': f'Hata: {str(e)}'}, status=400)


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
