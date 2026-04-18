"""palet_app.workers — background optimization worker & cancel/progress yardımcıları.

HTTP view'ları bu modülü ``run_optimization`` ile thread içinde çağırır.
Cancel registry'si process içindedir (tek-worker dev sunucuda yeterli).
"""

from .cancel_registry import (
    OptimizationCancelled,
    cancel_opt,
    cancel_group,
    is_cancelled,
    check_cancel,
)
from .progress import phase_progress, normalize_progress, estimate_mix_sec
from .optimization_worker import run_optimization, run_greedy_mix

__all__ = [
    "OptimizationCancelled",
    "cancel_opt",
    "cancel_group",
    "is_cancelled",
    "check_cancel",
    "phase_progress",
    "normalize_progress",
    "estimate_mix_sec",
    "run_optimization",
    "run_greedy_mix",
]
