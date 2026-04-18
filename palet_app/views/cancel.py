"""Cancel endpoint'leri — sayfadan ayrılma anında navigator.sendBeacon ile tetiklenir."""

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from ..workers import cancel_opt, cancel_group


@csrf_exempt
def cancel_optimization(request):
    """Tekil optimizasyon iptali (processing sayfasından ayrılma)."""
    opt_id = request.session.get('optimization_id')
    if opt_id:
        cancel_opt(opt_id)
        print(f"[IPTAL] Tekil optimization iptali istendi: {opt_id}")
    return JsonResponse({'success': True})


@csrf_exempt
def cancel_benchmark(request):
    """Benchmark grubu iptali (benchmark sayfasından ayrılma)."""
    group_id = request.session.get('benchmark_group_id')
    if group_id:
        cancel_group(group_id)
        print(f"[IPTAL] Benchmark grup iptali istendi: {group_id}")
    return JsonResponse({'success': True})
