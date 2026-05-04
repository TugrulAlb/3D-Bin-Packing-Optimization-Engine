"""Single palet yerleştirme servisi (Django entegrasyonu)."""

import logging

from src.core.single_pallet import (
    simulate_single_pallet,
    generate_grid_placement,
    DEFAULT_SINGLE_THRESHOLD,
)
from src.utils.helpers import group_products_smart

from .converters import django_urun_to_urundata, container_info_to_config
from ._palet_builder import create_django_palet


logger = logging.getLogger(__name__)


def single_palet_yerlestirme(urunler, container_info, optimization=None):
    """Single palet pipeline; Django modelleriyle çalışır, src/ algoritmasını kullanır.

    Returns:
        tuple: (single_pallets, yerlesmemis_urunler)
    """
    logger.info("Single palet operasyonu başlıyor")

    palet_cfg = container_info_to_config(container_info)

    all_products = [django_urun_to_urundata(urun) for urun in urunler]

    groups = group_products_smart(all_products)

    single_pallets = []
    mix_pool = []
    total_palet_id = 1

    for key, group_items in groups.items():
        urun_kodu = key[0]
        total_qty = len(group_items)

        logger.debug("Grup inceleniyor: %s, adet=%d", urun_kodu, total_qty)

        sim_result = simulate_single_pallet(group_items, palet_cfg)

        if sim_result["can_be_single"]:
            capacity = sim_result["capacity"]
            efficiency = sim_result["efficiency"]
            item_volume = group_items[0].boy * group_items[0].en * group_items[0].yukseklik
            pallet_volume = palet_cfg.volume

            if total_qty >= capacity:
                num_full_pallets = total_qty // capacity
                remainder = total_qty % capacity
                remainder_fill_ratio = (remainder * item_volume) / pallet_volume if remainder > 0 else 0
                create_partial = (remainder_fill_ratio >= DEFAULT_SINGLE_THRESHOLD)

                logger.info(
                    "Single onaylandı: %s eff=%.1f%% cap=%d full=%d remainder=%d",
                    urun_kodu, efficiency * 100, capacity, num_full_pallets, remainder,
                )
                if remainder > 0 and not create_partial:
                    logger.info("Single %s: remainder %d -> mix pool", urun_kodu, remainder)
            else:
                num_full_pallets = 0
                remainder = total_qty
                partial_fill_ratio = (remainder * item_volume) / pallet_volume
                create_partial = (partial_fill_ratio >= DEFAULT_SINGLE_THRESHOLD)

                if not create_partial:
                    logger.info(
                        "Single reddedildi: %s fill=%.1f%% < %.0f%%",
                        urun_kodu, partial_fill_ratio * 100, DEFAULT_SINGLE_THRESHOLD * 100,
                    )
                    mix_pool.extend(group_items)
                    continue

            for palet_no in range(num_full_pallets):
                palet_items = group_items[palet_no * capacity:(palet_no + 1) * capacity]
                placements = generate_grid_placement(palet_items, palet_cfg)

                palet = create_django_palet(
                    placements, palet_cfg, optimization, total_palet_id, 'single'
                )
                single_pallets.append(palet)
                total_palet_id += 1

            if create_partial and remainder > 0:
                palet_items = group_items[-remainder:]
                placements = generate_grid_placement(palet_items, palet_cfg)

                palet = create_django_palet(
                    placements, palet_cfg, optimization, total_palet_id, 'single'
                )
                single_pallets.append(palet)
                total_palet_id += 1
            elif remainder > 0:
                leftovers = group_items[-remainder:]
                mix_pool.extend(leftovers)
        else:
            logger.info("Single reddedildi: %s — %s", urun_kodu, sim_result["reason"])
            mix_pool.extend(group_items)

    logger.info(
        "Single tamamlandı: %d palet, mix havuzu: %d ürün",
        len(single_pallets), len(mix_pool),
    )

    yerlesmemis_urunler = []
    for item in mix_pool:
        urun_obj = next((u for u in urunler if u.id == item.id), None)
        if urun_obj:
            yerlesmemis_urunler.append(urun_obj)

    return single_pallets, yerlesmemis_urunler
