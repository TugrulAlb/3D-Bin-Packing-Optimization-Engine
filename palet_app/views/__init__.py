"""palet_app.views — HTTP view layer.

URL'ler ``config.urls``/``palet_app.urls`` üzerinden ``views.<name>``
biçiminde bu modüle bağlanır. Geriye dönük uyumluluk için tüm public isimler
burada re-export edilir; iç organizasyon alt modüllerde yapılır.
"""

from .upload import upload_result, urun_listesi, home_view
from .optimization import (
    processing,
    start_placement,
    optimization_status,
    analysis,
)
from .palet_detail import palet_detail, palet_3d_data
from .benchmark import (
    start_benchmark,
    benchmark_processing,
    benchmark_status,
    benchmark_result,
    benchmark_select,
)
from .cancel import cancel_optimization, cancel_benchmark
from ..workers import run_optimization, OptimizationCancelled

__all__ = [
    "upload_result",
    "urun_listesi",
    "home_view",
    "processing",
    "start_placement",
    "optimization_status",
    "analysis",
    "palet_detail",
    "palet_3d_data",
    "start_benchmark",
    "benchmark_processing",
    "benchmark_status",
    "benchmark_result",
    "benchmark_select",
    "cancel_optimization",
    "cancel_benchmark",
    "run_optimization",
    "OptimizationCancelled",
]
