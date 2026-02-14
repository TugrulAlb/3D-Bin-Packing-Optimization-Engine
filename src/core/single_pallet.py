"""
Single (Tek Tip) Palet Yerleştirme Algoritması
=================================================

Aynı boyuttaki ürünleri tek palete grid düzeninde yerleştirir.
%90+ efficiency threshold ile Single Palet onayı verir.

Özellikler:
    - Mixed-Orientation Tiling (Karma yönelim döşemesi)
    - Efficiency-Based Evaluation (Verimlilik bazlı değerlendirme)
    - Multi-Strategy Layer Configuration (Çok stratejili katman yapısı)
"""

from ..utils.helpers import urun_hacmi
from ..models.container import PaletConfig


def solve_best_layer_configuration(palet_L, palet_W, item_L, item_W):
    """
    Mixed-Orientation Tiling Algorithm.
    
    Palet tabanına sığabilecek maksimum ürün sayısını hesaplar.
    3 strateji dener:
        1. %100 Yönelim A (item_L × item_W)
        2. %100 Yönelim B (item_W × item_L) 
        3. Karma Yönelim (A satırları + B satırları)
    
    Args:
        palet_L: Palet uzunluğu
        palet_W: Palet genişliği
        item_L: Ürün uzunluğu
        item_W: Ürün genişliği
        
    Returns:
        tuple: (items_per_layer, layout_description, config_dict)
    """
    best_count = 0
    best_layout_desc = ""
    best_config = {'type_a_rows': 0, 'type_b_rows': 0, 'cols_a': 0, 'cols_b': 0, 'orientation': 0}
    
    # Strategy 1: 100% Orientation A
    cols_a = int(palet_L // item_L)
    rows_a_max = int(palet_W // item_W)
    
    if cols_a > 0 and rows_a_max > 0:
        count = cols_a * rows_a_max
        if count > best_count:
            best_count = count
            best_layout_desc = f"{rows_a_max} rows of {cols_a} items ({item_L}×{item_W})"
            best_config = {
                'type_a_rows': rows_a_max, 'type_b_rows': 0,
                'cols_a': cols_a, 'cols_b': 0, 'orientation': 0
            }
    
    # Strategy 2: 100% Orientation B
    cols_b = int(palet_L // item_W)
    rows_b_max = int(palet_W // item_L)
    
    if cols_b > 0 and rows_b_max > 0:
        count = cols_b * rows_b_max
        if count > best_count:
            best_count = count
            best_layout_desc = f"{rows_b_max} rows of {cols_b} items ({item_W}×{item_L})"
            best_config = {
                'type_a_rows': 0, 'type_b_rows': rows_b_max,
                'cols_a': 0, 'cols_b': cols_b, 'orientation': 1
            }
    
    # Strategy 3: Mixed-Orientation Tiling (A satırları + B satırları)
    if cols_a > 0 and cols_b > 0:
        for i in range(1, rows_a_max + 1):
            count_a = cols_a * i
            used_width_a = i * item_W
            remaining_width = palet_W - used_width_a
            rows_b_possible = int(remaining_width // item_L)
            count_b = cols_b * rows_b_possible
            total_count = count_a + count_b
            
            if total_count > best_count:
                best_count = total_count
                best_layout_desc = (f"{i} rows Type-A ({cols_a} items @ {item_L}×{item_W}) + "
                                   f"{rows_b_possible} rows Type-B ({cols_b} items @ {item_W}×{item_L})")
                best_config = {
                    'type_a_rows': i, 'type_b_rows': rows_b_possible,
                    'cols_a': cols_a, 'cols_b': cols_b, 'orientation': 2
                }
    
    # Strategy 3b: Reverse Mixed (B satırları + A satırları)
    if cols_a > 0 and cols_b > 0:
        for i in range(1, rows_b_max + 1):
            count_b = cols_b * i
            used_width_b = i * item_L
            remaining_width = palet_W - used_width_b
            rows_a_possible = int(remaining_width // item_W)
            count_a = cols_a * rows_a_possible
            total_count = count_a + count_b
            
            if total_count > best_count:
                best_count = total_count
                best_layout_desc = (f"{i} rows Type-B ({cols_b} items @ {item_W}×{item_L}) + "
                                   f"{rows_a_possible} rows Type-A ({cols_a} items @ {item_L}×{item_W})")
                best_config = {
                    'type_a_rows': rows_a_possible, 'type_b_rows': i,
                    'cols_a': cols_a, 'cols_b': cols_b, 'orientation': 3
                }
    
    return best_count, best_layout_desc, best_config


def generate_grid_placement(items_to_place, palet_cfg: PaletConfig):
    """
    Mixed-Orientation Grid Placement.
    
    Ürünler için X, Y, Z koordinatları üretir.
    Karma yönelim döşemesini destekler.
    
    Args:
        items_to_place: UrunData listesi (aynı tip)
        palet_cfg: PaletConfig nesnesi
        
    Returns:
        list[dict]: Her biri {'urun', 'x', 'y', 'z', 'L', 'W', 'H'} içeren yerleşim
    """
    if not items_to_place:
        return []
    
    u0 = items_to_place[0]
    PL, PW, PH = palet_cfg.length, palet_cfg.width, palet_cfg.height
    
    items_per_layer, layout_desc, layer_config = solve_best_layer_configuration(
        PL, PW, u0.boy, u0.en
    )
    
    if items_per_layer == 0:
        return []
    
    item_H = u0.yukseklik
    max_layers = int(PH // item_H)
    
    placements = []
    item_idx = 0
    
    type_a_rows = layer_config['type_a_rows']
    type_b_rows = layer_config['type_b_rows']
    cols_a = layer_config['cols_a']
    cols_b = layer_config['cols_b']
    orientation = layer_config['orientation']
    
    if orientation in [0, 2, 3]:
        item_L_a, item_W_a = u0.boy, u0.en
    else:
        item_L_a, item_W_a = u0.en, u0.boy
    
    if orientation in [1, 2, 3]:
        item_L_b, item_W_b = u0.en, u0.boy
    else:
        item_L_b, item_W_b = u0.boy, u0.en
    
    for layer in range(max_layers):
        z = layer * item_H
        y_offset = 0
        
        for row in range(type_a_rows):
            y = y_offset + row * item_W_a
            for col in range(cols_a):
                if item_idx >= len(items_to_place):
                    return placements
                x = col * item_L_a
                placements.append({
                    'urun': items_to_place[item_idx],
                    'x': x, 'y': y, 'z': z,
                    'L': item_L_a, 'W': item_W_a, 'H': item_H
                })
                item_idx += 1
        
        y_offset += type_a_rows * item_W_a
        
        for row in range(type_b_rows):
            y = y_offset + row * item_W_b
            for col in range(cols_b):
                if item_idx >= len(items_to_place):
                    return placements
                x = col * item_L_b
                placements.append({
                    'urun': items_to_place[item_idx],
                    'x': x, 'y': y, 'z': z,
                    'L': item_L_b, 'W': item_W_b, 'H': item_H
                })
                item_idx += 1
    
    return placements


def simulate_single_pallet(urun_listesi, palet_cfg: PaletConfig):
    """
    Efficiency-Based Single Pallet Simulation.
    
    Ürün grubunun Single Palet'e uygun olup olmadığını değerlendirir.
    %90+ verimlilik eşiği kullanır.
    
    Args:
        urun_listesi: Aynı tipteki UrunData listesi
        palet_cfg: PaletConfig nesnesi
        
    Returns:
        dict: {
            'can_be_single': bool,
            'capacity': int,
            'pack_count': int,
            'efficiency': float,
            'layout_desc': str,
            'reason': str
        }
    """
    if not urun_listesi:
        return {
            "can_be_single": False, "capacity": 0, "pack_count": 0,
            "efficiency": 0, "layout_desc": "", "reason": "Empty product list"
        }
    
    u0 = urun_listesi[0]
    PL, PW, PH = palet_cfg.length, palet_cfg.width, palet_cfg.height
    max_w = palet_cfg.max_weight
    
    # 1. Optimal katman konfigürasyonu
    items_per_layer, layout_desc, layer_config = solve_best_layer_configuration(
        PL, PW, u0.boy, u0.en
    )
    
    if items_per_layer == 0:
        return {
            "can_be_single": False, "capacity": 0, "pack_count": 0,
            "efficiency": 0, "layout_desc": "No valid configuration",
            "reason": "Product dimensions exceed pallet size"
        }
    
    # 2. Maksimum katman sayısı
    max_layers = int(PH // u0.yukseklik)
    if max_layers == 0:
        return {
            "can_be_single": False, "capacity": 0, "pack_count": 0,
            "efficiency": 0, "layout_desc": layout_desc,
            "reason": "Product height exceeds pallet height"
        }
    
    # 3. Hacim bazlı kapasite
    capacity_by_volume = items_per_layer * max_layers
    
    # 4. Ağırlık bazlı kapasite
    capacity_by_weight = int(max_w / u0.agirlik) if u0.agirlik > 0 else 999999
    
    # 5. Final kapasite
    capacity = min(capacity_by_volume, capacity_by_weight)
    
    # 6. Verimlilik hesabı
    item_volume = urun_hacmi(u0)
    pallet_volume = palet_cfg.volume
    efficiency = (capacity * item_volume) / pallet_volume
    
    # 7. %90 eşik kararı
    is_suitable = (efficiency >= 0.90)
    
    # 8. Mevcut stok için pack_count
    current_stock = len(urun_listesi)
    pack_count = min(current_stock, capacity)
    
    # 9. Detaylı açıklama
    if is_suitable:
        if capacity_by_volume < capacity_by_weight:
            constraint = "volume-limited"
        elif capacity_by_weight < capacity_by_volume:
            constraint = "weight-limited"
        else:
            constraint = "perfectly balanced"
        reason = (f"✅ Efficiency: {efficiency*100:.1f}% ({constraint}) | "
                 f"Capacity: {capacity} items | Layout: {layout_desc}")
    else:
        reason = (f"❌ Efficiency: {efficiency*100:.1f}% < 90% threshold | "
                 f"Capacity: {capacity} items | Layout: {layout_desc}")
    
    return {
        "can_be_single": is_suitable,
        "capacity": capacity,
        "pack_count": pack_count,
        "efficiency": efficiency,
        "layout_desc": layout_desc,
        "reason": reason
    }
