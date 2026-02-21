import logging
from .packing import *

def pack_maximal_rectangles_first_fit(urunler, palet_cfg, min_support_ratio=0.75, debug_support=False):
    if debug_support:
        print(f"[PACK] support_check_enabled={min_support_ratio > 0} min_support_ratio={min_support_ratio:.2f}")
    
    support_check_count = 0
    support_reject_prints = 0
    support_reject_total = 0
    total_support_checks = 0
    max_debug_prints = 20
    EPS = 1e-6
    
    open_pallets = []
    
    for idx, urun in enumerate(urunler):
        u_wgt = urun.agirlik
        orientations = possible_orientations_for(urun)
        
        best_pallet_idx = -1
        best_rect = None
        best_orientation = None
        best_score = float('inf')
        
        # Try to fit in existing open pallets (First-Fit)
        for p_idx, pallet in enumerate(open_pallets):
            if pallet['weight'] + u_wgt > palet_cfg.max_weight:
                continue
                
            for dims in orientations:
                item_l, item_w, item_h = dims
                for rect in pallet['free_rects']:
                    if rect.can_fit(item_l, item_w, item_h):
                        support_layer = pallet['layer_map'].get(snap_z(rect.z), [])
                        support_ratio = compute_support_ratio(
                            candidate_x=rect.x, candidate_y=rect.y, candidate_z=rect.z,
                            candidate_l=item_l, candidate_w=item_w,
                            placed_items=pallet['items'], layer_items=support_layer, debug=False
                        )
                        
                        if support_ratio + EPS < min_support_ratio:
                            continue
                            
                        if rect.z > EPS_Z:
                            n_corners, max_oh = compute_corner_support(
                                rect.x, rect.y, rect.z, item_l, item_w, support_layer
                            )
                            corner_ok = (n_corners >= MIN_SUPPORTED_CORNERS)
                            overhang_ok = (max_oh <= MAX_OVERHANG_CM)
                            if not (corner_ok and overhang_ok):
                                if CORNER_HARD_REJECT:
                                    continue
                        
                        residual_l = rect.length - item_l
                        residual_w = rect.width - item_w
                        short_side = min(residual_l, residual_w)
                        
                        # YENİ AMAZON MANTIĞI (Deepest-Bottom-Left):
                        # 1. Öncelik: En aşağısı (Z) - Havada uçmayı ve U boşlukları engeller
                        # 2. Öncelik: En arka (Y)
                        # 3. Öncelik: En sol (X)
                        # 4. Öncelik: En sıkı oturan (short_side)
                        score = (rect.z * 100000) + (rect.y * 1000) + (rect.x * 10) + short_side
                        
                        if score < best_score:
                            best_score = score
                            best_rect = rect
                            best_orientation = (item_l, item_w, item_h)
                            best_pallet_idx = p_idx
            
            # If we found a fit in this pallet, we can stop searching other pallets (First-Fit)
            if best_rect is not None:
                break
                
        # If it didn't fit in any open pallet, create a new one
        if best_rect is None:
            new_pallet = {
                'items': [],
                'weight': 0.0,
                'layer_map': {},
                'free_rects': [FreeRectangle(0, 0, 0, palet_cfg.length, palet_cfg.width, palet_cfg.height)]
            }
            open_pallets.append(new_pallet)
            best_pallet_idx = len(open_pallets) - 1
            
            for dims in orientations:
                item_l, item_w, item_h = dims
                rect = new_pallet['free_rects'][0]
                if rect.can_fit(item_l, item_w, item_h):
                    residual_l = rect.length - item_l
                    residual_w = rect.width - item_w
                    short_side = min(residual_l, residual_w)
                    
                    # Yeni palet için de aynı puanlama
                    score = (rect.z * 100000) + (rect.y * 1000) + (rect.x * 10) + short_side
                    if score < best_score:
                        best_score = score
                        best_rect = rect
                        best_orientation = (item_l, item_w, item_h)
                        
            if best_rect is None:
                raise ValueError(f"Item '{urun.urun_kodu}' cannot fit into an empty pallet")
                
        # Place the item in the chosen pallet
        target_pallet = open_pallets[best_pallet_idx]
        u_l, u_w, u_h = best_orientation
        placed_x, placed_y = best_rect.x, best_rect.y
        placed_z = snap_to_layer_z(best_rect.z, target_pallet['layer_map'])
        
        target_pallet['items'].append({
            'urun': urun, 'x': placed_x, 'y': placed_y, 'z': placed_z,
            'L': u_l, 'W': u_w, 'H': u_h
        })
        target_pallet['weight'] += u_wgt
        layer_key = snap_z(placed_z + u_h)
        target_pallet['layer_map'].setdefault(layer_key, []).append(target_pallet['items'][-1])
        
        new_free_rects = []
        for rect in target_pallet['free_rects']:
            if intersects_3d(rect, placed_x, placed_y, placed_z, u_l, u_w, u_h):
                sub_rects = split_rectangle_maximal(rect, placed_x, placed_y, placed_z, u_l, u_w, u_h)
                new_free_rects.extend(sub_rects)
            else:
                new_free_rects.append(rect)
                
        target_pallet['free_rects'] = remove_redundant_rectangles(new_free_rects)
        
    return [{'items': p['items'], 'weight': p['weight']} for p in open_pallets if p['items']]
