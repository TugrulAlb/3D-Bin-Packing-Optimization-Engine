"""Tekil optimizasyon başlatma, ilerleme ve analiz view'ları."""

import json
from threading import Thread

from django.db import transaction
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse

from ..models import Optimization, Palet, PHASE_RANGES
from ..services import ozet_grafikler_olustur
from ..workers import run_optimization, normalize_progress


def processing(request):
    """İşlem simülasyonu sayfası."""
    if 'urun_verileri' not in request.session:
        print("UYARI: processing: Session'da urun_verileri yok!")
        return redirect('palet_app:home')

    container_info = request.session.get('container_info')
    if not container_info:
        print("UYARI: processing: Session'da container_info yok!")
        return redirect('palet_app:home')

    optimization_id = request.session.get('optimization_id')
    print(f"processing sayfasi yuklendi (Optimization ID: {optimization_id})")

    return render(request, 'palet_app/processing.html')


def start_placement(request):
    """Yerleştirme işlemini başlatır."""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Yalnızca POST istekleri kabul edilir.'}, status=400)

    if 'urun_verileri' not in request.session:
        return JsonResponse({'success': False, 'error': 'Ürün verileri bulunamadı.'}, status=400)

    container_info = request.session.get('container_info')
    if not container_info:
        return JsonResponse({'success': False, 'error': 'Container bilgisi bulunamadı.'}, status=400)

    algoritma_raw = None
    ga_mode = 'balanced'
    try:
        body = {}
        if request.body:
            body = json.loads(request.body)
            algoritma_raw = body.get('algoritma')
        if algoritma_raw is None:
            algoritma_raw = request.POST.get('algoritma')
        if algoritma_raw is not None:
            algoritma = str(algoritma_raw).strip().lower()
        else:
            algoritma = 'genetic'
        if algoritma not in ('genetic', 'differential_evolution', 'greedy'):
            print(f"[start_placement] Gecersiz algoritma '{algoritma_raw}' -> genetic kullaniliyor")
            algoritma = 'genetic'
        if algoritma == 'genetic':
            gm = (body.get('ga_mode') or 'balanced').strip().lower()
            if gm in ('fast', 'balanced', 'quality'):
                ga_mode = gm
        print(f"[start_placement] Algoritma: {algoritma!r}, GA mod: {ga_mode!r}")
    except Exception as e:
        print(f"[start_placement] Body parse hatasi, varsayilan genetic kullaniliyor: {e}")
        algoritma = 'genetic'

    container_length = container_info.get('length', 120)
    container_width = container_info.get('width', 100)
    container_height = container_info.get('height', 180)
    container_weight = container_info.get('weight', 1250)

    with transaction.atomic():
        optimization = Optimization.objects.create(
            palet_tipi=None,
            container_length=container_length,
            container_width=container_width,
            container_height=container_height,
            container_weight=container_weight,
            algoritma=algoritma,
            islem_durumu=json.dumps({
                "current_step": 0,
                "total_steps": 5,
                "messages": [],
            }),
        )

        request.session['optimization_id'] = optimization.id
        request.session['algoritma'] = algoritma
        request.session.modified = True

        print(f"\n{'='*60}")
        print("YENI OPTIMIZASYON BASLATILDI")
        print(f"{'='*60}")
        print(f"Optimization ID: {optimization.id}")
        print(f"Algoritma: {algoritma}")
        print(f"Container: {container_length}x{container_width}x{container_height} cm")
        print(f"Max Ağırlık: {container_weight} kg")
        print(f"Ürün Sayısı: {len(request.session['urun_verileri'])}")
        print(f"{'='*60}\n")

        container_dict = {
            'length': container_length,
            'width': container_width,
            'height': container_height,
            'weight': container_weight,
        }

        try:
            thread = Thread(
                target=run_optimization,
                args=(request.session['urun_verileri'], container_dict, optimization.id, algoritma, ga_mode),
            )
            thread.daemon = True
            thread.start()
            print(f"Thread baslatildi (ID: {optimization.id})")
        except Exception as e:
            print(f"HATA: Thread baslatma hatasi: {str(e)}")
            import traceback
            traceback.print_exc()
            return JsonResponse({
                'success': False,
                'error': f'Thread başlatılamadı: {str(e)}',
            }, status=500)

    return JsonResponse({
        'success': True,
        'message': 'Optimizasyon başlatıldı.',
        'optimization_id': optimization.id,
        'status_url': reverse('palet_app:optimization_status'),
    })


def optimization_status(request):
    """Optimizasyon durumunu döndürür."""
    optimization_id = request.session.get('optimization_id')
    if not optimization_id:
        print("UYARI: optimization_status: Session'da optimization_id yok!")
        return JsonResponse({'success': False, 'error': 'Optimizasyon bulunamadı.'}, status=400)

    try:
        optimization = Optimization.objects.get(id=optimization_id)
        durum = optimization.get_islem_durumu()

        print(f"Status check (ID: {optimization_id}): Completed={optimization.tamamlandi}, Step={durum.get('current_step', 0)}/{durum.get('total_steps', 5)}")

        if optimization.tamamlandi:
            print("Optimization tamamlandi, yonlendirme yapiliyor...")
            return JsonResponse({
                'success': True,
                'completed': True,
                'next_url': reverse('palet_app:analysis'),
            })

        is_error = durum.get('current_step', 0) == -1
        cur, tot, pct = normalize_progress(
            durum.get('current_step', 0),
            durum.get('total_steps', 5),
            completed=False,
            durum=durum,
            error=is_error,
        )
        phase_label = PHASE_RANGES.get(durum.get('phase', ''), (0, 0, 'Hazırlanıyor'))[2]
        return JsonResponse({
            'success': True,
            'completed': False,
            'current_step': cur,
            'total_steps': tot,
            'percent': pct,
            'phase_label': phase_label,
            'messages': durum.get('messages', []),
        })

    except Optimization.DoesNotExist:
        print(f"Optimization bulunamadi (ID: {optimization_id})")
        return JsonResponse({'success': False, 'error': 'Optimizasyon bulunamadı.'}, status=400)


def analysis(request):
    """Optimizasyon sonuçlarını gösterir."""
    optimization_id = request.session.get('optimization_id')
    if not optimization_id:
        print("UYARI: analysis: Session'da optimization_id yok!")
        return redirect('palet_app:home')

    print(f"analysis view cagrildi (ID: {optimization_id})")

    try:
        optimization = get_object_or_404(Optimization, id=optimization_id)

        print(f"   Tamamlandı: {optimization.tamamlandi}")
        print(f"   Toplam Palet: {optimization.toplam_palet}")

        if not optimization.tamamlandi:
            print("UYARI: Optimizasyon henuz tamamlanmamis, processing'e yonlendiriliyor...")
            return redirect('palet_app:processing')

        paletler = Palet.objects.filter(optimization=optimization).order_by('palet_id')
        print(f"   Bulunan palet sayısı: {paletler.count()}")

        pie_chart_html, bar_chart_html = ozet_grafikler_olustur(optimization)

        benchmark_siblings = []
        if optimization.benchmark_group_id:
            sibling_qs = Optimization.objects.filter(
                benchmark_group_id=optimization.benchmark_group_id,
                tamamlandi=True,
            ).exclude(id=optimization.id).order_by('id')
            algo_label = {
                'greedy': 'Greedy',
                'genetic': 'Genetik Algoritma',
                'differential_evolution': 'Differential Evolution',
            }
            for sib in sibling_qs:
                elapsed = None
                if sib.bitis_zamani and sib.baslangic_zamani:
                    elapsed = round((sib.bitis_zamani - sib.baslangic_zamani).total_seconds(), 2)
                benchmark_siblings.append({
                    'id': sib.id,
                    'algoritma': sib.algoritma,
                    'label': algo_label.get(sib.algoritma, sib.algoritma),
                    'toplam_palet': sib.toplam_palet,
                    'elapsed': elapsed,
                })

        context = {
            'optimization': optimization,
            'paletler': paletler,
            'single_oran': optimization.single_palet / optimization.toplam_palet * 100 if optimization.toplam_palet > 0 else 0,
            'mix_oran': optimization.mix_palet / optimization.toplam_palet * 100 if optimization.toplam_palet > 0 else 0,
            'yerlesmemis_urunler': optimization.yerlesmemis_urunler,
            'pie_chart_html': pie_chart_html,
            'bar_chart_html': bar_chart_html,
            'benchmark_siblings': benchmark_siblings,
        }

        print("Analysis sayfasi render ediliyor...")
        return render(request, 'palet_app/analysis.html', context)

    except Optimization.DoesNotExist:
        print(f"Optimization bulunamadi (ID: {optimization_id})")
        return redirect('palet_app:home')
