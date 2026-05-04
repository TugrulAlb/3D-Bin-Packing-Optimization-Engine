"""Tekil optimizasyon başlatma, ilerleme ve analiz view'ları."""

import json
import logging
from threading import Thread

from django.db import transaction
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse

from ..models import Optimization, Palet, PHASE_RANGES
from ..services import ozet_grafikler_olustur
from ..workers import run_optimization, normalize_progress


logger = logging.getLogger(__name__)


def processing(request):
    """İşlem simülasyonu sayfası."""
    if 'urun_verileri' not in request.session:
        return redirect('palet_app:home')

    container_info = request.session.get('container_info')
    if not container_info:
        return redirect('palet_app:home')

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
            logger.warning("Geçersiz algoritma '%s', genetic kullanılıyor", algoritma_raw)
            algoritma = 'genetic'
        if algoritma == 'genetic':
            gm = (body.get('ga_mode') or 'balanced').strip().lower()
            if gm in ('fast', 'balanced', 'quality'):
                ga_mode = gm
    except (json.JSONDecodeError, ValueError, AttributeError) as e:
        logger.warning("Body parse hatası, varsayılan genetic kullanılıyor: %s", e)
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

        logger.info(
            "Yeni optimizasyon başlatıldı: id=%d algo=%s ga_mode=%s container=%sx%sx%s w=%s items=%d",
            optimization.id, algoritma, ga_mode,
            container_length, container_width, container_height, container_weight,
            len(request.session['urun_verileri']),
        )

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
                daemon=True,
            )
            thread.start()
        except Exception as e:
            logger.exception("Thread başlatma hatası (opt %s)", optimization.id)
            return JsonResponse({
                'success': False,
                'error': 'Thread başlatılamadı.',
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
        return JsonResponse({'success': False, 'error': 'Optimizasyon bulunamadı.'}, status=400)

    try:
        optimization = Optimization.objects.get(id=optimization_id)
        durum = optimization.get_islem_durumu()

        if optimization.tamamlandi:
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
        return JsonResponse({'success': False, 'error': 'Optimizasyon bulunamadı.'}, status=400)


def analysis(request):
    """Optimizasyon sonuçlarını gösterir."""
    optimization_id = request.session.get('optimization_id')
    if not optimization_id:
        return redirect('palet_app:home')

    try:
        optimization = get_object_or_404(Optimization, id=optimization_id)

        if not optimization.tamamlandi:
            return redirect('palet_app:processing')

        paletler = Palet.objects.filter(optimization=optimization).order_by('palet_id')

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

        return render(request, 'palet_app/analysis.html', context)

    except Optimization.DoesNotExist:
        return redirect('palet_app:home')
