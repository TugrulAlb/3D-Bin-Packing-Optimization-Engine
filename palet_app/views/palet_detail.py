"""Palet detay ve 3D veri endpoint'leri."""

from django.http import JsonResponse, HttpResponseBadRequest
from django.shortcuts import render, redirect, get_object_or_404

from ..models import Urun, Palet, Optimization
from src.utils.visualization import renk_uret


def palet_detail(request, palet_id):
    """Tek bir palet detayını gösterir."""
    optimization_id = request.session.get('optimization_id')
    if not optimization_id:
        return redirect('palet_app:home')

    try:
        optimization = get_object_or_404(Optimization, id=optimization_id)

        if not optimization.tamamlandi:
            return redirect('palet_app:processing')

        palet = get_object_or_404(Palet, optimization=optimization, palet_id=palet_id)

        tum_paletler = Palet.objects.filter(optimization=optimization).order_by('palet_id')
        palet_ids = list(tum_paletler.values_list('palet_id', flat=True))

        current_index = palet_ids.index(palet_id)
        prev_id = palet_ids[current_index - 1] if current_index > 0 else None
        next_id = palet_ids[current_index + 1] if current_index < len(palet_ids) - 1 else None

        urun_konumlari = palet.json_to_dict(palet.urun_konumlari)
        urun_boyutlari = palet.json_to_dict(palet.urun_boyutlari)

        urun_ids = [int(id) for id in urun_konumlari.keys()]
        urunler = list(Urun.objects.filter(id__in=urun_ids))

        urun_renkleri = {}
        for urun in urunler:
            if urun.urun_kodu not in urun_renkleri:
                urun_renkleri[urun.urun_kodu] = renk_uret(urun.urun_kodu)

        urun_detaylari = []
        for urun in urunler:
            konum = urun_konumlari.get(str(urun.id), [0, 0, 0])
            boyut = urun_boyutlari.get(str(urun.id), [0, 0, 0])

            if isinstance(konum, list):
                konum = tuple(konum)
            if isinstance(boyut, list):
                boyut = tuple(boyut)

            renk_rgb = urun_renkleri.get(urun.urun_kodu, (0.5, 0.5, 0.5))
            renk_rgb_255 = (int(renk_rgb[0] * 255), int(renk_rgb[1] * 255), int(renk_rgb[2] * 255))

            urun_detaylari.append({
                'urun': urun,
                'konum': konum,
                'boyut': boyut,
                'renk_rgb': renk_rgb_255,
            })

        context = {
            'palet': palet,
            'urun_detaylari': urun_detaylari,
            'prev_id': prev_id,
            'next_id': next_id,
            'total_palets': len(palet_ids),
        }

        return render(request, 'palet_app/palet_detail.html', context)

    except Exception as e:
        return HttpResponseBadRequest(f"Hata: {str(e)}")


def palet_3d_data(request, palet_id):
    """3D görselleştirme için palet verisini JSON formatında döndürür."""
    optimization_id = request.session.get('optimization_id')
    if not optimization_id:
        return JsonResponse({'error': 'Optimizasyon bulunamadı'}, status=400)

    try:
        optimization = get_object_or_404(Optimization, id=optimization_id)
        palet = get_object_or_404(Palet, optimization=optimization, palet_id=palet_id)

        palet_data = {
            'palet_id': palet.palet_id,
            'boy': palet.boy,
            'en': palet.en,
            'yukseklik': palet.max_yukseklik,
            'doluluk': palet.doluluk_orani(),
            'agirlik': float(palet.toplam_agirlik),
            'items': [],
        }

        urun_konumlari = palet.json_to_dict(palet.urun_konumlari)
        urun_boyutlari = palet.json_to_dict(palet.urun_boyutlari)
        urun_ids = [int(id) for id in urun_konumlari.keys()]
        urunler = {urun.id: urun for urun in Urun.objects.filter(id__in=urun_ids)}

        for urun_id_str, konum in urun_konumlari.items():
            urun_id = int(urun_id_str)
            if urun_id not in urunler:
                continue

            urun = urunler[urun_id]
            boyut = urun_boyutlari.get(urun_id_str, [0, 0, 0])

            renk_rgb = renk_uret(urun.urun_kodu)

            palet_data['items'].append({
                'urun_kodu': urun.urun_kodu,
                'urun_adi': urun.urun_adi,
                'x': konum[0],
                'y': konum[1],
                'z': konum[2],
                'boy': boyut[0],
                'en': boyut[1],
                'yukseklik': boyut[2],
                'agirlik': float(urun.agirlik),
                'renk': {
                    'r': renk_rgb[0],
                    'g': renk_rgb[1],
                    'b': renk_rgb[2],
                },
            })

        return JsonResponse(palet_data)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
