"""Toplu test (benchmark) view'ları — 3 algoritmayı sıralı çalıştırır."""

import json
import uuid
from threading import Thread

from django.db import transaction
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.urls import reverse

from ..models import Optimization, Palet, PHASE_RANGES  # noqa: F401 (PHASE_RANGES legacy re-export)
from ..workers import run_optimization, is_cancelled, normalize_progress


BENCHMARK_ALGORITHMS = [
    {'key': 'greedy', 'label': 'Greedy', 'ga_mode': None},
    {'key': 'genetic', 'label': 'Genetik Algoritma', 'ga_mode': 'balanced'},
    {'key': 'differential_evolution', 'label': 'Differential Evolution', 'ga_mode': None},
]


def _algoritma_label(key):
    return {
        'greedy': 'Greedy',
        'genetic': 'Genetik Algoritma',
        'differential_evolution': 'Differential Evolution',
    }.get(key, key)


def start_benchmark(request):
    """3 algoritmayı aynı ürün verisi üzerinde SIRAYLA çalıştırır."""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Yalnızca POST istekleri kabul edilir.'}, status=400)

    if 'urun_verileri' not in request.session:
        return JsonResponse({'success': False, 'error': 'Ürün verileri bulunamadı.'}, status=400)

    container_info = request.session.get('container_info')
    if not container_info:
        return JsonResponse({'success': False, 'error': 'Container bilgisi bulunamadı.'}, status=400)

    container_length = container_info.get('length', 120)
    container_width = container_info.get('width', 100)
    container_height = container_info.get('height', 180)
    container_weight = container_info.get('weight', 1250)

    group_id = str(uuid.uuid4())
    created_ids = []

    with transaction.atomic():
        for algo in BENCHMARK_ALGORITHMS:
            optimization = Optimization.objects.create(
                palet_tipi=None,
                container_length=container_length,
                container_width=container_width,
                container_height=container_height,
                container_weight=container_weight,
                algoritma=algo['key'],
                benchmark_group_id=group_id,
                islem_durumu=json.dumps({
                    "current_step": 0,
                    "total_steps": 5,
                    "messages": [],
                }),
            )
            created_ids.append((optimization.id, algo['key'], algo['ga_mode']))

    request.session['benchmark_group_id'] = group_id
    request.session.modified = True

    print(f"\n{'='*60}\nBENCHMARK BASLATILDI (Group: {group_id})\n{'='*60}")

    container_dict = {
        'length': container_length,
        'width': container_width,
        'height': container_height,
        'weight': container_weight,
    }

    urun_verileri_snapshot = list(request.session['urun_verileri'])

    def _run_benchmark_sequence():
        for opt_id, algo_key, ga_mode in created_ids:
            if is_cancelled(group_id=group_id):
                print("[Benchmark] Grup iptal edildi, kalanlar atlaniyor.")
                try:
                    remaining = Optimization.objects.filter(
                        benchmark_group_id=group_id, tamamlandi=False
                    )
                    for r in remaining:
                        d = r.get_islem_durumu()
                        if d.get('current_step', 0) != -1:
                            r.islem_adimi_ekle("İşlem kullanıcı tarafından iptal edildi.")
                            d = r.get_islem_durumu()
                            d['current_step'] = -1
                            d['cancelled'] = True
                            r.islem_durumu = json.dumps(d)
                            r.save()
                except Exception as _e:
                    print(f"Iptal temizlik hatasi: {_e}")
                break
            print(f"[Benchmark] BASLIYOR: {algo_key} (ID: {opt_id})")
            try:
                run_optimization(
                    urun_verileri_snapshot,
                    container_dict,
                    opt_id,
                    algo_key,
                    ga_mode or 'balanced',
                    group_id,
                )
                print(f"[Benchmark] BITTI: {algo_key} (ID: {opt_id})")
            except Exception as e:
                print(f"HATA: Benchmark ({algo_key}): {e}")

    try:
        t = Thread(target=_run_benchmark_sequence)
        t.daemon = True
        t.start()
        print("[Benchmark] Sirali dispatcher thread baslatildi.")
    except Exception as e:
        print(f"HATA: Benchmark dispatcher baslatma: {e}")

    return JsonResponse({
        'success': True,
        'benchmark_group_id': group_id,
        'status_url': reverse('palet_app:benchmark_status'),
        'processing_url': reverse('palet_app:benchmark_processing'),
    })


def benchmark_processing(request):
    """Benchmark ilerleme sayfası."""
    group_id = request.session.get('benchmark_group_id')
    if not group_id:
        return redirect('palet_app:home')

    optimizations = Optimization.objects.filter(benchmark_group_id=group_id).order_by('id')
    if not optimizations.exists():
        return redirect('palet_app:home')

    return render(request, 'palet_app/benchmark_processing.html', {
        'benchmark_group_id': group_id,
        'optimizations': optimizations,
    })


def benchmark_status(request):
    """Grup içindeki tüm optimizasyonların durumunu tek JSON'da döndürür."""
    group_id = request.session.get('benchmark_group_id')
    if not group_id:
        return JsonResponse({'success': False, 'error': 'Benchmark grubu bulunamadı.'}, status=400)

    optimizations = Optimization.objects.filter(benchmark_group_id=group_id).order_by('id')

    items = []
    all_done = True
    any_error = False
    for opt in optimizations:
        durum = opt.get_islem_durumu()
        is_error = durum.get('current_step', 0) == -1
        cur, tot, pct = normalize_progress(
            durum.get('current_step', 0),
            durum.get('total_steps', 5),
            completed=opt.tamamlandi,
            durum=durum,
            error=is_error,
        )
        if is_error:
            any_error = True
        if not opt.tamamlandi and not is_error:
            all_done = False

        messages = durum.get('messages', []) or []
        elapsed = None
        if opt.bitis_zamani and opt.baslangic_zamani:
            elapsed = round((opt.bitis_zamani - opt.baslangic_zamani).total_seconds(), 2)

        items.append({
            'id': opt.id,
            'algoritma': opt.algoritma,
            'completed': opt.tamamlandi,
            'error': is_error,
            'current_step': cur,
            'total_steps': tot,
            'percent': pct,
            'last_message': messages[-1] if messages else '',
            'messages': messages,
            'elapsed': elapsed,
        })

    return JsonResponse({
        'success': True,
        'all_done': all_done,
        'any_error': any_error,
        'items': items,
        'result_url': reverse('palet_app:benchmark_result'),
    })


def benchmark_result(request):
    """3 algoritmanın sonuçlarını yan yana karşılaştırır."""
    group_id = request.session.get('benchmark_group_id')
    if not group_id:
        return redirect('palet_app:home')

    optimizations = Optimization.objects.filter(benchmark_group_id=group_id).order_by('id')
    if not optimizations.exists():
        return redirect('palet_app:home')

    palet_cfg_dims = None
    results = []
    for opt in optimizations:
        paletler = Palet.objects.filter(optimization=opt)

        fill_ratios = []
        total_used_vol = 0.0
        container_vol = max(1.0, (opt.container_length or 1) * (opt.container_width or 1) * (opt.container_height or 1))

        for p in paletler:
            boyutlar = p.json_to_dict(p.urun_boyutlari)
            used = 0.0
            for _id, dims in boyutlar.items():
                if isinstance(dims, (list, tuple)) and len(dims) >= 3:
                    used += float(dims[0]) * float(dims[1]) * float(dims[2])
            total_used_vol += used
            fill_ratios.append(used / container_vol * 100.0)

        avg_fill = (sum(fill_ratios) / len(fill_ratios)) if fill_ratios else 0.0
        elapsed = None
        if opt.bitis_zamani and opt.baslangic_zamani:
            elapsed = (opt.bitis_zamani - opt.baslangic_zamani).total_seconds()

        if palet_cfg_dims is None:
            palet_cfg_dims = {
                'length': opt.container_length,
                'width': opt.container_width,
                'height': opt.container_height,
                'weight': opt.container_weight,
            }

        results.append({
            'optimization': opt,
            'algoritma': opt.algoritma,
            'algoritma_label': _algoritma_label(opt.algoritma),
            'completed': opt.tamamlandi,
            'toplam_palet': opt.toplam_palet,
            'single_palet': opt.single_palet,
            'mix_palet': opt.mix_palet,
            'avg_fill': round(avg_fill, 2),
            'elapsed': round(elapsed, 2) if elapsed is not None else None,
            'yerlesmemis_sayi': len(opt.yerlesmemis_urunler or []),
        })

    completed_results = [r for r in results if r['completed'] and r['toplam_palet'] > 0]
    best_id = None
    if completed_results:
        best = min(completed_results, key=lambda r: (r['toplam_palet'], -r['avg_fill']))
        best_id = best['optimization'].id
    for r in results:
        r['is_best'] = (r['optimization'].id == best_id)

    return render(request, 'palet_app/benchmark_result.html', {
        'results': results,
        'container': palet_cfg_dims,
        'benchmark_group_id': group_id,
    })


def benchmark_select(request, optimization_id):
    """Benchmark sonuçlarından birini seç → analysis sayfasına yönlendir."""
    group_id = request.session.get('benchmark_group_id')
    try:
        opt = Optimization.objects.get(id=optimization_id, benchmark_group_id=group_id)
    except Optimization.DoesNotExist:
        return redirect('palet_app:home')

    request.session['optimization_id'] = opt.id
    request.session['algoritma'] = opt.algoritma
    request.session.modified = True
    return redirect('palet_app:analysis')
