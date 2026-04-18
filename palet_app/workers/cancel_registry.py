"""Process-içi iptal kayıtları.

Kullanıcı ``processing`` ya da ``benchmark_processing`` sayfasından ayrılınca
frontend ``navigator.sendBeacon`` ile ilgili endpoint'e ping atar; endpoint
buradaki ``cancel_opt``/``cancel_group`` çağırır. Worker thread'i ise
fazlar arası ``check_cancel`` ile bu seti kontrol eder.
"""

import threading


_CANCEL_LOCK = threading.Lock()
_CANCELLED_OPTS: set[int] = set()
_CANCELLED_GROUPS: set[str] = set()


class OptimizationCancelled(Exception):
    """Worker'a iptal sinyali taşır."""
    pass


def cancel_opt(opt_id) -> None:
    with _CANCEL_LOCK:
        _CANCELLED_OPTS.add(int(opt_id))


def cancel_group(group_id) -> None:
    if not group_id:
        return
    with _CANCEL_LOCK:
        _CANCELLED_GROUPS.add(str(group_id))


def is_cancelled(opt_id=None, group_id=None) -> bool:
    with _CANCEL_LOCK:
        if opt_id is not None and int(opt_id) in _CANCELLED_OPTS:
            return True
        if group_id is not None and str(group_id) in _CANCELLED_GROUPS:
            return True
    return False


def check_cancel(opt_id, group_id=None) -> None:
    if is_cancelled(opt_id=opt_id, group_id=group_id):
        raise OptimizationCancelled()
