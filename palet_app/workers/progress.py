"""Faz tabanlı ilerleme yüzdesi hesabı (ease-out).

``Optimization.set_phase('mix', expected_sec=...)`` faza girildiğinde
``PHASE_RANGES`` içindeki (start, end, label) değerlerini kullanarak UI
tarafındaki progress bar'ın yumuşak ilerlemesini besler.
"""

import math
import time

from ..models import PHASE_RANGES


def phase_progress(durum, completed: bool = False, error: bool = False):
    """
    - Her fazın PHASE_RANGES içinde (start, end, label) değeri vardır.
    - Faza girildiğinde yüzde = start; fazın tahmini süresi boyunca
      ease-out (1 - e^(-t/tau)) eğrisiyle (end - 1)'e yaklaşır.
    - Asla end'e ulaşmaz; bir sonraki set_phase() ile yüzde start'a çıkar.
    - completed=True → 100. error → mevcut yüzde korunur (kırmızı bar).
    - Monotonluk garantisi: durum['last_pct'] ile yüzde asla geri gitmez.
    """
    if completed:
        return 100, 'Tamamlandı'

    phase = durum.get('phase')
    last_pct = float(durum.get('last_pct', 0))

    if not phase or phase not in PHASE_RANGES:
        return int(round(last_pct)), durum.get('phase_label', 'Hazırlanıyor')

    start, end, label = PHASE_RANGES[phase]
    start = float(start)
    end = float(end)
    span = max(0.0, end - start - 1.0)  # end'e asla dokunmasın

    expected = max(0.5, float(durum.get('phase_expected_sec', 5.0)))
    elapsed = max(0.0, time.time() - float(durum.get('phase_start', time.time())))
    tau = expected / 3.0  # ~3*tau sonunda %95 doygunluk

    k = 1.0 - math.exp(-elapsed / tau) if tau > 0 else 1.0
    pct = start + span * k

    pct = max(pct, last_pct)  # geri gitme
    pct = max(0.0, min(99.0, pct))
    return int(round(pct)), label


def normalize_progress(current_step, total_steps, completed: bool = False, durum=None, error: bool = False):
    """current_step/total_steps + faz tabanlı yüzde. ``durum`` verilmişse faz kullanılır."""
    total = max(1, int(total_steps))
    cur = max(0, min(int(current_step), total))
    if durum is None:
        pct = 100 if completed else max(0, min(95, int(round(100 * cur / total))))
        return cur, total, pct
    pct, _label = phase_progress(durum, completed=completed, error=error)
    return cur, total, pct


def estimate_mix_sec(algoritma: str, urun_sayisi: int) -> float:
    """Mix aşamasının tahmini süresi (saniye). Kaba — ease-out zaten tolerant."""
    n = max(1, int(urun_sayisi))
    if algoritma == 'greedy':
        return max(3.0, 2.0 + n * 0.05)
    if algoritma == 'differential_evolution':
        return max(15.0, 15.0 + n * 0.6)
    return max(10.0, 10.0 + n * 0.8)
