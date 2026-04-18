"""Single palet yerleştirme servisi (Django entegrasyonu)."""

from src.core.single_pallet import (
    simulate_single_pallet,
    generate_grid_placement,
    DEFAULT_SINGLE_THRESHOLD,
)
from src.utils.helpers import group_products_smart

from .converters import django_urun_to_urundata, container_info_to_config
from ._palet_builder import create_django_palet


def single_palet_yerlestirme(urunler, container_info, optimization=None):
    """
    Single Palet sürecini yöneten ana fonksiyon.
    Django modelleriyle çalışır, src/ algoritmasını kullanır.

    Returns:
        tuple: (single_pallets, yerlesmemis_urunler)
    """
    print("--- Single Palet Operasyonu Başlıyor ---")

    palet_cfg = container_info_to_config(container_info)

    all_products = [django_urun_to_urundata(urun) for urun in urunler]

    groups = group_products_smart(all_products)

    single_pallets = []
    mix_pool = []
    total_palet_id = 1

    for key, group_items in groups.items():
        urun_kodu = key[0]
        total_qty = len(group_items)

        print(f"Grup İnceleniyor: {urun_kodu}, Adet: {total_qty}")

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

                print(f"  -> ONAYLANDI. {sim_result['reason']}")
                print(f"  -> Efficiency: {efficiency*100:.1f}% | Capacity: {capacity} items/pallet")
                print(f"  -> Stock: {total_qty} → {num_full_pallets} full pallet(s)")

                if remainder > 0:
                    if create_partial:
                        print(f"  -> + 1 partial pallet ({remainder} items, fill: {remainder_fill_ratio*100:.1f}%)")
                    else:
                        print(f"  -> {remainder} remainder items -> Mix Pool")
            else:
                num_full_pallets = 0
                remainder = total_qty
                partial_fill_ratio = (remainder * item_volume) / pallet_volume
                create_partial = (partial_fill_ratio >= DEFAULT_SINGLE_THRESHOLD)

                if create_partial:
                    print(f"  -> ONAYLANDI (Partial). Fill: {partial_fill_ratio*100:.1f}%")
                else:
                    print(f"  -> REJECTED. Fill: {partial_fill_ratio*100:.1f}% < {DEFAULT_SINGLE_THRESHOLD:.0%}")
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
                print(f"  -> {remainder} items sent to Mix Pool")
        else:
            print(f"  -> REDDEDILDI. {sim_result['reason']}")
            mix_pool.extend(group_items)

    print(f"--- Single Bitti. {len(single_pallets)} palet. Mix Havuzu: {len(mix_pool)} ürün. ---")

    yerlesmemis_urunler = []
    for item in mix_pool:
        urun_obj = next((u for u in urunler if u.id == item.id), None)
        if urun_obj:
            yerlesmemis_urunler.append(urun_obj)

    return single_pallets, yerlesmemis_urunler
