"""JSON payload -> (urun_verileri, container_info) parser.

Hem web upload (palet_app.views.upload) hem de REST API (api.views) bu fonksiyonu
kullanır; davranış birebir aynıdır. Hata durumunda Türkçe mesajla `ValueError` fırlatır.
"""


MAX_DETAIL_ROWS = 5_000
MAX_EXPANDED_ITEMS = 500_000


def _to_float(x, default=0.0):
    try:
        return float(x) if x is not None else default
    except (TypeError, ValueError):
        return default


def parse_optimization_payload(payload):
    """Web upload ve API ortak parser'ı.

    Parametre:
        payload: dict — `{"id?", "container", "details": [...]}` veya doğrudan
                 `urun_verileri` listesi (legacy format).

    Döner:
        (urun_verileri: list[dict], container_info: dict)

    Hata:
        ValueError — geçersiz format / boş liste / eksik alan / DoS guard
    """
    if isinstance(payload, list):
        urun_verileri = payload
        container_info = {}
        if not urun_verileri:
            raise ValueError("Geçersiz JSON formatı. Ürün listesi boş veya hatalı.")
        _validate_required_fields(urun_verileri)
        return urun_verileri, container_info

    if not isinstance(payload, dict) or 'details' not in payload:
        raise ValueError('Geçersiz JSON formatı. Desteklenen format: {"details": [...]}')

    detaylar = payload.get('details', []) or []
    if len(detaylar) > MAX_DETAIL_ROWS:
        raise ValueError(f"details satır sayısı çok fazla (max {MAX_DETAIL_ROWS}).")

    container_info = dict(payload.get('container') or {})
    palet_id = payload.get('id')
    if palet_id is not None:
        container_info['palet_id'] = palet_id

    urun_verileri = []

    for detail in detaylar:
        if not isinstance(detail, dict):
            raise ValueError("details içindeki öğeler dict olmalı.")
        product = detail.get('product') or {}
        package_quantity = detail.get('package_quantity')
        quantity = detail.get('quantity', 0)
        unit_id = detail.get('unit_id', 'ADET')

        code = product.get('code', product.get('id', 'UNKNOWN'))

        package_length = _to_float(product.get('package_length'))
        package_width = _to_float(product.get('package_width'))
        package_height = _to_float(product.get('package_height'))
        package_weight = _to_float(product.get('package_weight'))

        unit_length = _to_float(product.get('unit_length'))
        unit_width = _to_float(product.get('unit_width'))
        unit_height = _to_float(product.get('unit_height'))
        unit_weight = _to_float(product.get('unit_weight'))

        mukavemet = _to_float(product.get('package_max_stack_weight'), default=100000)
        if mukavemet == 0:
            mukavemet = 100000

        if package_quantity is None or package_quantity <= 0:
            if unit_id == 'KG' and unit_weight > 0:
                adet_urun = int(quantity / unit_weight)
            else:
                adet_urun = int(quantity)

            if adet_urun > MAX_EXPANDED_ITEMS:
                raise ValueError(
                    f"Bir SKU için adet patlaması: {adet_urun} > {MAX_EXPANDED_ITEMS}"
                )
            if len(urun_verileri) + adet_urun > MAX_EXPANDED_ITEMS:
                raise ValueError(
                    f"Toplam ürün sayısı çok fazla (max {MAX_EXPANDED_ITEMS})."
                )

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
                    'quantity': _to_float(quantity),
                    'unit_length': unit_length,
                    'unit_width': unit_width,
                    'unit_height': unit_height,
                    'unit_weight': unit_weight,
                })
        else:
            pq = int(package_quantity)
            if pq > MAX_EXPANDED_ITEMS:
                raise ValueError(
                    f"package_quantity çok büyük: {pq} > {MAX_EXPANDED_ITEMS}"
                )
            if len(urun_verileri) + pq > MAX_EXPANDED_ITEMS:
                raise ValueError(
                    f"Toplam ürün sayısı çok fazla (max {MAX_EXPANDED_ITEMS})."
                )
            for _ in range(pq):
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
                    'package_quantity': pq,
                    'quantity': _to_float(quantity),
                    'unit_length': unit_length,
                    'unit_width': unit_width,
                    'unit_height': unit_height,
                    'unit_weight': unit_weight,
                })

    if not urun_verileri:
        raise ValueError("Geçersiz JSON formatı. Ürün listesi boş veya hatalı.")

    _validate_required_fields(urun_verileri)
    return urun_verileri, container_info


def _validate_required_fields(urun_verileri):
    required_fields = ['urun_kodu', 'urun_adi', 'boy', 'en', 'yukseklik', 'agirlik']
    for urun in urun_verileri:
        if not isinstance(urun, dict):
            raise ValueError("Ürün öğesi dict olmalı.")
        for field in required_fields:
            if field not in urun:
                raise ValueError(f"Eksik alan: {field}")
