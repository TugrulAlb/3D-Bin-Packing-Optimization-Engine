"""Merge & Repack post-optimizasyon servisleri.

İki yaklaşım: rastgele restart (v2) ve iteratif BFD (mix).
"""

import logging

from ._palet_builder import create_django_palet


logger = logging.getLogger(__name__)


def _django_pallets_to_packer_dicts(mix_paletler, urun_data_by_id, log_prefix: str):
    packer_pallets = []
    for palet in mix_paletler:
        konumlar = palet.json_to_dict(palet.urun_konumlari)
        boyutlar = palet.json_to_dict(palet.urun_boyutlari)
        items = []
        total_weight = 0.0
        for urun_id_str, pos in konumlar.items():
            urun_id = int(urun_id_str)
            ud = urun_data_by_id.get(urun_id)
            if ud is None:
                logger.warning("%s UrunData id=%s bulunamadı, atlanıyor", log_prefix, urun_id_str)
                continue
            dim = boyutlar.get(urun_id_str, [ud.boy, ud.en, ud.yukseklik])
            items.append({
                'urun': ud,
                'x': float(pos[0]),
                'y': float(pos[1]),
                'z': float(pos[2]),
                'L': float(dim[0]),
                'W': float(dim[1]),
                'H': float(dim[2]),
            })
            total_weight += ud.agirlik
        packer_pallets.append({'items': items, 'weight': total_weight})
    return packer_pallets


def merge_repack_service(mix_paletler, palet_cfg, optimization, baslangic_id, urun_data_listesi):
    """Random Restart Merge & Repack (v2) servisi.

    Sonuç kabul edilmezse orijinal paletler değiştirilmeden döner.
    """
    from src.core.merge_repack import merge_and_repack_v2, MergeRepackMetrics

    if len(mix_paletler) < 2:
        return mix_paletler, MergeRepackMetrics.no_op("insufficient_pallets")

    urun_data_by_id = {ud.id: ud for ud in urun_data_listesi}
    packer_pallets = _django_pallets_to_packer_dicts(mix_paletler, urun_data_by_id, "[MR]")

    optimized, metrics = merge_and_repack_v2(packer_pallets, palet_cfg)

    if not metrics.accepted:
        return mix_paletler, metrics

    for palet in mix_paletler:
        palet.delete()

    new_django_paletler = []
    palet_id = baslangic_id
    for pallet_data in optimized:
        p = create_django_palet(
            pallet_data['items'],
            palet_cfg,
            optimization,
            palet_id,
            'mix',
            items_are_dicts=True,
        )
        new_django_paletler.append(p)
        palet_id += 1

    logger.info(
        "[MR] Service complete: %d → %d mix pallets. %s",
        len(mix_paletler), len(new_django_paletler), metrics.summary(),
    )
    return new_django_paletler, metrics


def merge_repack_mix_service(mix_paletler, palet_cfg, optimization, baslangic_id, urun_data_listesi):
    """İteratif BFD Merge & Repack servisi.

    Sonuç kabul edilmezse orijinal liste, DB değiştirilmeden döner.
    """
    from src.core.merge_repack import merge_and_repack_mix, MixMergeMetrics

    if len(mix_paletler) < 2:
        return mix_paletler, MixMergeMetrics.no_op("insufficient_pallets")

    urun_data_by_id = {ud.id: ud for ud in urun_data_listesi}
    packer_pallets = _django_pallets_to_packer_dicts(mix_paletler, urun_data_by_id, "[MixMerge]")

    optimized, metrics = merge_and_repack_mix(packer_pallets, palet_cfg)

    if not metrics.accepted:
        return mix_paletler, metrics

    for palet in mix_paletler:
        palet.delete()

    new_django_paletler = []
    palet_id = baslangic_id
    for pallet_data in optimized:
        p = create_django_palet(
            pallet_data['items'],
            palet_cfg,
            optimization,
            palet_id,
            'mix',
            items_are_dicts=True,
        )
        new_django_paletler.append(p)
        palet_id += 1

    logger.info(
        "[MixMerge] Service done: %d → %d pallets. %s",
        len(mix_paletler), len(new_django_paletler), metrics.summary(),
    )
    return new_django_paletler, metrics
