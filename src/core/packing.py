"""
3D Maximal Rectangles Packing Engine
======================================

Endüstriyel seviye 3D kutu yerleştirme algoritmaları.

Algoritmalar:
    - Maximal Rectangles (Ana motor) - Auto-Orientation destekli
    - Shelf-Based Packing (Legacy destek)

Referanslar:
    - Jylänki, J. "A Thousand Ways to Pack the Bin" (2010)
    - Huang, E. & Korf, R. "Optimal Rectangle Packing" (2013)
"""

import logging
import math

from ..utils.helpers import possible_orientations_for

logger = logging.getLogger(__name__)


# ====================================================================
# LAYER SNAPPING KONFİGÜRASYONU  (Amazon-like gerçekçilik)
# ====================================================================
# Z_GRID: Katman hizalaması için grid adımı (cm).
# Ürünlerin z konumları bu değerin katlarına yuvarlanır (mümkünse).
# Düşürüldükçe daha hassas katman hizalaması sağlar (ama daha yavaş değil).
Z_GRID   = 5.0    # cm – konfig için bu sabitini değiştirin
EPS_Z    = 1e-6   # z-snap için float hassasiyet toleransı

# Yeni palet açma log'u için referans doluluk eşiği
_LOG_MIN_UTIL = 0.45   # Bu seviyenin altında palet kapanırsa ⚠️ underfill uyarısı verilir

# ---------------------------------------------------------------
# KÖŞE DESTEK (CORNER SUPPORT) KONFIGÜRASYONU
# ---------------------------------------------------------------
# MIN_SUPPORTED_CORNERS : 4 köşenün en az kaçı desteklenmeli (0-4)
MIN_SUPPORTED_CORNERS = 3       # Amazon-like: 3/4 köşe desteklenmeli
# MAX_OVERHANG_CM       : desteksiz köşeye izin verilen max taşma (cm)
MAX_OVERHANG_CM       = 8.0    # Bu mesafeyi aşan yerleşim reddedilir
# CORNER_HARD_REJECT    : True = hard reject, False = sadece ceza uygulanır
CORNER_HARD_REJECT    = True
# W_CORNER_PENALTY      : soft-reject modunda fitness ceza çarpanı (HARD=False ise)
W_CORNER_PENALTY      = 5000.0


def snap_z(z: float, eps: float = EPS_Z) -> float:
    """
    Kanonik Z normalizeştirici.

    Her yerde aynı float gösterimine dönüştürür; layer_map anahtar
    çakışmalarını (float drift) önler.

    Kural: layer_map'e yazılan veya okunan her z değeri bu
    fonksiyondan geçirilmelidir.
    """
    return round(z, 6)


def compute_corner_support(
    cx: float, cy: float, cz: float,
    cl: float, cw: float,
    layer_items: list,
    tol: float = EPS_Z
) -> tuple:
    """
    Köşe Destek Kontrolü (Corner Support Check).

    Yerleştirilmek istenen kutunun 4 alt köşesini kontrol eder:
    Her köşe, alt katmandaki herhangi bir kutunun alanı içinde
    mı düşüyor?

    Ayrıca: köşe desteksizse, desteksizliğin taşma mesafesini ölçer.
    Amaç: çıkıntı (bridge/overhang) oluşumunu önlemek.

    Args:
        cx, cy, cz: Adayın konum koordinatları (alt-sol-ön köşe)
        cl, cw:     Adayın boy ve enı
        layer_items: Alt katmandaki kutular (snap_z(cz) ile filtrelenmiş)
        tol:        Tolerans (varsayılan EPS_Z)

    Returns:
        (supported_count: int,   # 0-4 arası; kaç köşe destekleniyor
         max_overhang_cm: float  # desteksiz köşenin maksimum taşma mesafesi)
    """
    if cz <= tol:
        # Zemin üzeri: destek garantili
        return 4, 0.0

    corners = [
        (cx,      cy),       # sol-ön
        (cx + cl, cy),       # sağ-ön
        (cx,      cy + cw),  # sol-arka
        (cx + cl, cy + cw),  # sağ-arka
    ]

    supported_count = 0
    max_overhang    = 0.0

    for (px, py) in corners:
        corner_supported = False
        min_dist_to_support = float('inf')

        for item in layer_items:
            ix1, ix2 = item['x'], item['x'] + item['L']
            iy1, iy2 = item['y'], item['y'] + item['W']

            if (ix1 - tol) <= px <= (ix2 + tol) and (iy1 - tol) <= py <= (iy2 + tol):
                corner_supported = True
                min_dist_to_support = 0.0
                break   # Bu köşe destekli; diğer kutulara bakmaya gerek yok

            # Desteksizse: merkeze olan mesafeyi hesapla (taşma proxy)
            dx = max(0.0, ix1 - px, px - ix2)
            dy = max(0.0, iy1 - py, py - iy2)
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < min_dist_to_support:
                min_dist_to_support = dist

        if corner_supported:
            supported_count += 1
        else:
            # Desteksiz köşe: taşma mesafesini güncelle
            overhang = min_dist_to_support if math.isfinite(min_dist_to_support) else MAX_OVERHANG_CM + 1
            if overhang > max_overhang:
                max_overhang = overhang

    return supported_count, max_overhang


def snap_to_layer_z(placed_z: float, layer_map: dict) -> float:
    """
    Layer Snapping: placed_z'yi mevcut katman sınırlarına hizalar.

    Adımlar:
      1. layer_map'te EPS_Z toleransı içinde bir anahtar varsa → o değeri kullan
         (floating point kaymalarını düzeltir, raf gibi katman hizalaması sağlar).
      2. Yoksa Z_GRID'e yuvarlama yap; ama sadece YUKARI (aşağı yuvarlamak
         alttaki kutuyla çakışmaya yol açar).
      3. Yukarı snap mesafesi Z_GRID/2'yi geçiyorsa → orijinal değeri koru.

    Args:
        placed_z:  Packing motoru tarafından belirlenen z koordinatı.
        layer_map: {round(z+h, 6): [items]} sözlüğü.

    Returns:
        float: Snap uygulanmış (ya da orijinal) z değeri.
    """
    # --- ADIM 1: Mevcut layer anahtarına yakın mı? ---
    for key in layer_map:
        if abs(key - placed_z) <= EPS_Z:
            return key   # Tam hizalama (float drift düzeltmesi)

    # --- ADIM 2: Z_GRID grid'ine yukari snap ---
    grid_below  = math.floor(placed_z / Z_GRID) * Z_GRID
    grid_above  = math.ceil(placed_z  / Z_GRID) * Z_GRID

    # Tam eşleşme (zaten grid üzerinde)
    if abs(placed_z - grid_below) <= EPS_Z:
        return grid_below
    if abs(placed_z - grid_above) <= EPS_Z:
        return grid_above

    # SADECE YUKARI snap: Aşağıya snap yaparsak kutu çakışır.
    # Maksimum snap mesafesini Z_GRID/2 ile sınırla.
    snap_up = grid_above
    if (snap_up - placed_z) <= Z_GRID / 2.0:
        return snap_up

    # --- ADIM 3: Snap uygulanamıyor, orijinali koru ---
    return placed_z


# ====================================================================
# GRAVITY / STABILITY CONSTRAINT
# ====================================================================

def compute_support_ratio(
    candidate_x: float,
    candidate_y: float,
    candidate_z: float,
    candidate_l: float,
    candidate_w: float,
    placed_items: list,
    layer_items: list = None,
    tol: float = 1e-6,
    debug: bool = False
) -> float:
    """
    AMAZON-STYLE STABILITY: Compute support ratio for a candidate placement.
    
    Support ratio = (total supported overlap area) / (base area of candidate box)
    
    A box placed at height z > 0 must have at least 40% of its base area
    supported by boxes directly underneath (z-level contact).
    
    Algorithm:
        1. Find all boxes whose top surface is at candidate's bottom (z == candidate_z)
        2. For each supporting box, compute XY overlap area between:
           - Candidate base: [candidate_x, candidate_x+candidate_l] × [candidate_y, candidate_y+candidate_w]
           - Support top: [sx, sx+sL] × [sy, sy+sW]
        3. Sum all overlap areas (multi-support allowed)
        4. Return total_overlap / base_area
    
    Args:
        candidate_x, candidate_y, candidate_z: Position of candidate box (bottom-left-front corner)
        candidate_l, candidate_w: Length and width of candidate box base
        placed_items: List of already placed boxes [{'x', 'y', 'z', 'L', 'W', 'H'}, ...]
        layer_items: Optional pre-filtered list of boxes at the support z-level. If provided,
                     z-level filtering is skipped for performance (O(1) vs O(n)).
        tol: Tolerance for z-level matching (default 1e-6)
        debug: Enable debug logging (default False)
    
    Returns:
        float: Support ratio in [0.0, 1.0] (capped at 1.0)
    """
    if candidate_z <= tol:
        return 1.0
    
    base_area = candidate_l * candidate_w
    if base_area <= 0:
        return 0.0
    
    c_x1, c_x2 = candidate_x, candidate_x + candidate_l
    c_y1, c_y2 = candidate_y, candidate_y + candidate_w
    
    total_overlap_area = 0.0
    supporting_boxes = []
    
    scan_items = layer_items if layer_items is not None else placed_items
    
    for item in scan_items:
        if layer_items is None:
            support_top_z = item['z'] + item['H']
            if abs(support_top_z - candidate_z) > tol:
                continue
        
        s_x1, s_x2 = item['x'], item['x'] + item['L']
        s_y1, s_y2 = item['y'], item['y'] + item['W']
        
        overlap_length = max(0.0, min(c_x2, s_x2) - max(c_x1, s_x1))
        overlap_width  = max(0.0, min(c_y2, s_y2) - max(c_y1, s_y1))
        overlap_area = overlap_length * overlap_width
        
        if overlap_area > tol:
            total_overlap_area += overlap_area
            supporting_boxes.append({'box': item, 'overlap_area': overlap_area})
    
    total_overlap_area = min(total_overlap_area, base_area)
    support_ratio = total_overlap_area / base_area
    
    if debug and candidate_z > tol:
        print(f"  [SUPPORT CHECK] z={candidate_z:.2f}, base_area={base_area:.2f}, "
              f"overlap={total_overlap_area:.2f}, ratio={support_ratio:.2%}, "
              f"supports={len(supporting_boxes)}")
    
    return support_ratio


# ====================================================================
# VERİ YAPILARI
# ====================================================================

class FreeRectangle:
    """
    Boş dikdörtgen alanı temsil eder (Maximal Rectangles için).
    
    Attributes:
        x, y, z: Sol-alt-ön köşe koordinatları
        length, width, height: Boş alanın boyutları
        volume: Boş alanın hacmi
    """
    
    def __init__(self, x, y, z, length, width, height):
        self.x = x
        self.y = y
        self.z = z
        self.length = length
        self.width = width
        self.height = height
        self.volume = length * width * height
    
    def can_fit(self, item_l, item_w, item_h):
        """Ürün bu alana sığar mı?"""
        return (self.length >= item_l and 
                self.width >= item_w and 
                self.height >= item_h)
    
    def __repr__(self):
        return f"Rect({self.x},{self.y},{self.z} | {self.length}×{self.width}×{self.height})"


# ====================================================================
# 3D MAXIMAL RECTANGLES ALGORİTMASI
# ====================================================================

def intersects_3d(rect, placed_x, placed_y, placed_z, placed_l, placed_w, placed_h):
    """
    Boş dikdörtgenin yerleştirilen kutuyla 3D kesişimini kontrol eder.
    
    Returns:
        bool: Kesişim varsa True
    """
    return not (
        rect.x >= placed_x + placed_l or
        placed_x >= rect.x + rect.length or
        rect.y >= placed_y + placed_w or
        placed_y >= rect.y + rect.width or
        rect.z >= placed_z + placed_h or
        placed_z >= rect.z + rect.height
    )


def split_rectangle_maximal(rect, placed_x, placed_y, placed_z, placed_l, placed_w, placed_h):
    """
    TRUE 3D MAXIMAL RECTANGLES SPLITTING.
    
    Bir kutu yerleştirildiğinde kesişen boş dikdörtgeni en fazla 6 yeni
    alt-dikdörtgene böler:
    - Sol, Sağ (X ekseni)
    - Ön, Arka (Y ekseni)
    - Alt, Üst (Z ekseni)
    
    Bu ÖRTÜŞEN dikdörtgenler oluşturur - Maximal Rectangles'ın temel özelliği.
    
    Returns:
        list[FreeRectangle]: Yeni boş dikdörtgenler (0-6 arası)
    """
    new_rects = []
    
    # LEFT: Yerleştirilen kutunun solundaki alan
    if rect.x < placed_x:
        new_rects.append(FreeRectangle(
            rect.x, rect.y, rect.z,
            placed_x - rect.x,
            rect.width,
            rect.height
        ))
    
    # RIGHT: Yerleştirilen kutunun sağındaki alan
    if placed_x + placed_l < rect.x + rect.length:
        new_rects.append(FreeRectangle(
            placed_x + placed_l, rect.y, rect.z,
            (rect.x + rect.length) - (placed_x + placed_l),
            rect.width,
            rect.height
        ))
    
    # FRONT: Yerleştirilen kutunun önündeki alan
    if rect.y < placed_y:
        new_rects.append(FreeRectangle(
            rect.x, rect.y, rect.z,
            rect.length,
            placed_y - rect.y,
            rect.height
        ))
    
    # BACK: Yerleştirilen kutunun arkasındaki alan
    if placed_y + placed_w < rect.y + rect.width:
        new_rects.append(FreeRectangle(
            rect.x, placed_y + placed_w, rect.z,
            rect.length,
            (rect.y + rect.width) - (placed_y + placed_w),
            rect.height
        ))
    
    # BOTTOM: Yerleştirilen kutunun altındaki alan
    if rect.z < placed_z:
        new_rects.append(FreeRectangle(
            rect.x, rect.y, rect.z,
            rect.length,
            rect.width,
            placed_z - rect.z
        ))
    
    # TOP: Yerleştirilen kutunun üstündeki alan
    if placed_z + placed_h < rect.z + rect.height:
        new_rects.append(FreeRectangle(
            rect.x, rect.y, placed_z + placed_h,
            rect.length,
            rect.width,
            (rect.z + rect.height) - (placed_z + placed_h)
        ))
    
    return new_rects


def find_best_rectangle(free_rects, item_l, item_w, item_h):
    """
    Best Short Side Fit (BSSF) Heuristic.
    
    Yerleştirme sonrası minimum kısa kenar artığı oluşturacak
    boş dikdörtgeni seçer. İnce, kullanılamaz boşlukları engeller.
    
    Returns:
        FreeRectangle or None
    """
    best_rect = None
    min_short_side_residual = float('inf')
    
    for rect in free_rects:
        if rect.can_fit(item_l, item_w, item_h):
            residual_l = rect.length - item_l
            residual_w = rect.width - item_w
            short_side_residual = min(residual_l, residual_w)
            
            if short_side_residual < min_short_side_residual:
                min_short_side_residual = short_side_residual
                best_rect = rect
            elif short_side_residual == min_short_side_residual and best_rect is not None:
                current_vol_diff = rect.volume - (item_l * item_w * item_h)
                best_vol_diff = best_rect.volume - (item_l * item_w * item_h)
                if current_vol_diff < best_vol_diff:
                    best_rect = rect
    
    return best_rect


def remove_redundant_rectangles(rects):
    """
    Birbirinin içinde olan dikdörtgenleri kaldırır.
    Küçük olanı sil, büyüğü tut (daha geniş arama alanı).
    """
    filtered = []
    
    for i, rect1 in enumerate(rects):
        is_contained = False
        
        for j, rect2 in enumerate(rects):
            if i == j:
                continue
            
            if (rect2.x <= rect1.x and 
                rect2.y <= rect1.y and 
                rect2.z <= rect1.z and
                rect2.x + rect2.length >= rect1.x + rect1.length and
                rect2.y + rect2.width >= rect1.y + rect1.width and
                rect2.z + rect2.height >= rect1.z + rect1.height):
                is_contained = True
                break
        
        if not is_contained:
            filtered.append(rect1)
    
    return filtered


def pack_maximal_rectangles(urunler, palet_cfg, min_support_ratio=0.40, debug_support=False):
    """
    TRUE 3D MAXIMAL RECTANGLES ALGORITHM with AUTO-ORIENTATION & GRAVITY CONSTRAINT.
    
    Ana yerleştirme motoru. Temel özellikler:
    1. Kesişim tabanlı bölme: Kutu yerleştirildiğinde tüm kesişen boş
       dikdörtgenler en fazla 6 alt-dikdörtgene bölünür.
    2. Örtüşen dikdörtgenler: Guillotine'den farklı olarak örtüşen boş
       alanlar tutulur, sadece tamamen kapsananlar silinir.
    3. Auto-Orientation: Her ürün için tüm yönelimler denenir.
    4. GRAVITY CONSTRAINT: Placements at z > 0 require min_support_ratio (default 40%)
       of base area supported by boxes directly below (Amazon-style stability).
    
    Complexity: O(n × r × f) - n: ürün, r: yönelim, f: boş dikdörtgen
    
    Args:
        urunler: GA'dan gelen ürün sıralaması
        palet_cfg: PaletConfig nesnesi
        min_support_ratio: Minimum support ratio for placements above ground (default 0.40)
        debug_support: Enable debug logging for support checks (default False)
        
    Returns:
        list[dict]: Her palet için {'items': [...], 'weight': float}
    """
    # HARD VERIFICATION: Support constraint status
    if debug_support:
        print(f"[PACK] support_check_enabled={min_support_ratio > 0} min_support_ratio={min_support_ratio:.2f}")
    
    # PRODUCTIONIZATION: Rate-limiting counters and numerical tolerance
    support_check_count = 0
    support_reject_prints = 0
    support_reject_total = 0
    total_support_checks = 0
    max_debug_prints = 20
    EPS = 1e-6  # Numerical tolerance for borderline cases
    
    pallets = []
    current_pallet = {
        'items': [],
        'weight': 0.0,
        'layer_map': {},
        'free_rects': [FreeRectangle(
            0, 0, 0, 
            palet_cfg.length, palet_cfg.width, palet_cfg.height
        )]
    }
    
    for idx, urun in enumerate(urunler):
        u_wgt = urun.agirlik
        
        # Ağırlık kontrolü - yeni palet gerekiyor mu?
        if current_pallet['weight'] + u_wgt > palet_cfg.max_weight:
            if current_pallet['items']:
                if logger.isEnabledFor(logging.DEBUG):
                    _cur_vol  = sum(i["L"]*i["W"]*i["H"] for i in current_pallet['items'])
                    _cur_util = _cur_vol / palet_cfg.volume
                    _warn = " ⚠️ underfill" if _cur_util < _LOG_MIN_UTIL else ""
                    logger.debug(
                        "[PALLET OPEN] reason=weight_overflow | palet=%d util=%.1f%%%s "
                        "| item=%s (%.1f kg)",
                        len(pallets) + 1, _cur_util * 100, _warn,
                        getattr(urun, 'urun_kodu', '?'), u_wgt,
                    )
                pallets.append({
                    'items': current_pallet['items'],
                    'weight': current_pallet['weight']
                })
            current_pallet = {
                'items': [],
                'weight': 0.0,
                'layer_map': {},
                'free_rects': [FreeRectangle(
                    0, 0, 0,
                    palet_cfg.length, palet_cfg.width, palet_cfg.height
                )]
            }
        
        # AUTO-ORIENTATION + GRAVITY CONSTRAINT: Tüm yönelimler × tüm boş dikdörtgenler
        best_rect = None
        best_orientation = None
        min_short_side = float('inf')
        orientations = possible_orientations_for(urun)
        
        for dims in orientations:
            item_l, item_w, item_h = dims
            
            for rect in current_pallet['free_rects']:
                if rect.can_fit(item_l, item_w, item_h):
                    # GRAVITY CHECK: Verify support ratio for placements above ground
                    # snap_z: layer_map anahtarlarını tutarlı normalize et (float drift önleme)
                    support_layer = current_pallet['layer_map'].get(snap_z(rect.z), [])
                    support_ratio = compute_support_ratio(
                        candidate_x=rect.x,
                        candidate_y=rect.y,
                        candidate_z=rect.z,
                        candidate_l=item_l,
                        candidate_w=item_w,
                        placed_items=current_pallet['items'],
                        layer_items=support_layer,
                        debug=False
                    )

                    # DEBUG LOG: Support check applied (rate-limited to first 20)
                    if rect.z > 1e-6:
                        total_support_checks += 1
                        if debug_support and support_check_count < max_debug_prints:
                            print(f"[SUPPORT CHECK #{total_support_checks}] z={rect.z:.2f} support={support_ratio:.2%} "
                                  f"req={min_support_ratio:.2%} box={urun.urun_kodu}")
                            support_check_count += 1

                    # REJECT placement if insufficient area support (with numerical tolerance)
                    if support_ratio + EPS < min_support_ratio:
                        support_reject_total += 1
                        if debug_support:
                            if support_reject_prints < 30 or support_reject_total % 200 == 0:
                                print(f"[SUPPORT REJECT #{support_reject_total}] z={rect.z:.2f} support={support_ratio:.2%} "
                                      f"req={min_support_ratio:.2%} box={urun.urun_kodu}")
                                support_reject_prints += 1
                        continue

                    # CORNER SUPPORT CHECK: kaldıraç (bridge/overhang) engelle
                    if rect.z > EPS_Z:
                        n_corners, max_oh = compute_corner_support(
                            rect.x, rect.y, rect.z,
                            item_l, item_w,
                            support_layer
                        )
                        # Kriter 1: yeterli köşe sayısı
                        corner_ok = (n_corners >= MIN_SUPPORTED_CORNERS)
                        # Kriter 2: desteksiz köşenin taşma mesafesi
                        overhang_ok = (max_oh <= MAX_OVERHANG_CM)

                        if not (corner_ok and overhang_ok):
                            if CORNER_HARD_REJECT:
                                if debug_support:
                                    print(f"[CORNER REJECT] z={rect.z:.2f} corners={n_corners}/4 "
                                          f"overhang={max_oh:.1f}cm box={urun.urun_kodu}")
                                continue   # Hard reject: bu pozisyonu atla
                            # Soft reject: devam et ama skor kaybı fitness'e yansır
                            # (fitness.py'de per-item penalty hesaplanır)
                    
                    residual_l = rect.length - item_l
                    residual_w = rect.width - item_w
                    short_side = min(residual_l, residual_w)
                    
                    if short_side < min_short_side:
                        min_short_side = short_side
                        best_rect = rect
                        best_orientation = (item_l, item_w, item_h)
        
        # Hiçbir yönelimde sığmadıysa yeni palet aç
        if best_rect is None:
            if current_pallet['items']:
                if logger.isEnabledFor(logging.DEBUG):
                    _cur_vol  = sum(i["L"]*i["W"]*i["H"] for i in current_pallet['items'])
                    _cur_util = _cur_vol / palet_cfg.volume
                    _warn = " ⚠️ underfill" if _cur_util < _LOG_MIN_UTIL else ""
                    logger.debug(
                        "[PALLET OPEN] reason=no_fit | palet=%d util=%.1f%%%s | item=%s",
                        len(pallets) + 1, _cur_util * 100, _warn,
                        getattr(urun, 'urun_kodu', '?'),
                    )
                pallets.append({
                    'items': current_pallet['items'],
                    'weight': current_pallet['weight']
                })
            
            current_pallet = {
                'items': [],
                'weight': 0.0,
                'layer_map': {},
                'free_rects': [FreeRectangle(
                    0, 0, 0,
                    palet_cfg.length, palet_cfg.width, palet_cfg.height
                )]
            }
            
            best_rect = None
            best_orientation = None
            min_short_side = float('inf')
            
            for dims in orientations:
                item_l, item_w, item_h = dims
                rect = current_pallet['free_rects'][0]
                
                if rect.can_fit(item_l, item_w, item_h):
                    # First item on new pallet is always at z=0 (ground), no support check needed
                    residual_l = rect.length - item_l
                    residual_w = rect.width - item_w
                    short_side = min(residual_l, residual_w)
                    
                    if short_side < min_short_side:
                        min_short_side = short_side
                        best_rect = rect
                        best_orientation = (item_l, item_w, item_h)
            
            if best_rect is None:
                pallets.append({
                    'items': current_pallet['items'],
                    'weight': current_pallet['weight']
                }) if current_pallet['items'] else None
                raise ValueError(
                    f"Item '{urun.urun_kodu}' cannot fit into an empty pallet "
                    f"(palet: {palet_cfg.length}×{palet_cfg.width}×{palet_cfg.height}, "
                    f"max_weight: {palet_cfg.max_weight})"
                )
        
        # Ürünü en iyi yönelimle yerleştir
        u_l, u_w, u_h = best_orientation
        placed_x, placed_y = best_rect.x, best_rect.y
        # LAYER SNAPPING: z değerini mevcut katman sınırına veya Z_GRID grid'ine hizala.
        # Sadece YUKARI snap yapılır; aşağı snap alttaki kutuyla çakışmaya yol açar.
        placed_z = snap_to_layer_z(best_rect.z, current_pallet['layer_map'])
        
        current_pallet['items'].append({
            'urun': urun,
            'x': placed_x,
            'y': placed_y,
            'z': placed_z,
            'L': u_l,
            'W': u_w,
            'H': u_h
        })
        current_pallet['weight'] += u_wgt
        # snap_z: layer_map anahtarını kanonik forma dönüştür
        layer_key = snap_z(placed_z + u_h)
        current_pallet['layer_map'].setdefault(layer_key, []).append(current_pallet['items'][-1])
        
        # TRUE MAXIMAL RECTANGLES SPLITTING
        new_free_rects = []
        
        for rect in current_pallet['free_rects']:
            if intersects_3d(rect, placed_x, placed_y, placed_z, u_l, u_w, u_h):
                sub_rects = split_rectangle_maximal(
                    rect, placed_x, placed_y, placed_z, u_l, u_w, u_h
                )
                new_free_rects.extend(sub_rects)
            else:
                new_free_rects.append(rect)
        
        current_pallet['free_rects'] = new_free_rects
        current_pallet['free_rects'] = remove_redundant_rectangles(
            current_pallet['free_rects']
        )
    
    # Son paleti ekle
    if current_pallet['items']:
        pallets.append({
            'items': current_pallet['items'],
            'weight': current_pallet['weight']
        })
    
    # PRODUCTIONIZATION: Summary logging
    if debug_support and (total_support_checks > 0 or support_reject_total > 0):
        reject_rate = support_reject_total / max(1, total_support_checks) if total_support_checks > 0 else 0
        print(f"[SUPPORT SUMMARY] total_checks={total_support_checks} total_rejects={support_reject_total} "
              f"reject_rate={reject_rate:.1%}")
    
    return pallets


# ====================================================================
# SHELF-BASED PACKING (Legacy Destek)
# ====================================================================

def pack_shelf_based(urunler, rot_gen, palet_cfg, min_support_ratio=0.40, debug_support=False):
    """
    GA Motoru için Shelf (Raf) yerleştirme - Legacy (with Gravity Constraint).
    
    Args:
        urunler: Ürün listesi
        rot_gen: Rotasyon genleri (her ürün için yönelim indeksi)
        palet_cfg: PaletConfig nesnesi
        min_support_ratio: Minimum support ratio for placements above ground (default 0.40)
        debug_support: Enable debug logging for support checks (default False)
    """
    pallets = []
    current_items = []
    
    x, y, z = 0.0, 0.0, 0.0
    current_weight = 0.0
    current_shelf_height = 0.0
    current_shelf_y = 0.0    
    
    L, W, H = palet_cfg.length, palet_cfg.width, palet_cfg.height
    
    for idx, urun in enumerate(urunler):
        r = 0
        if rot_gen and idx < len(rot_gen):
            r = rot_gen[idx]
        
        dims = possible_orientations_for(urun)
        if r >= len(dims):
            r = 0
        u_l, u_w, u_h = dims[r]
        u_wgt = urun.agirlik
        
        if current_weight + u_wgt > palet_cfg.max_weight:
            if current_items:
                pallets.append({"items": current_items, "weight": current_weight})
            current_items = []
            current_weight = 0.0
            x, y, z = 0.0, 0.0, 0.0
            current_shelf_height, current_shelf_y = 0.0, 0.0

        if x + u_l > L:
            x = 0
            y += current_shelf_y if current_shelf_y > 0 else u_w
            current_shelf_y = 0 
            
        if y + u_w > W:
            x = 0
            y = 0
            z += current_shelf_height if current_shelf_height > 0 else u_h
            current_shelf_height = 0
            
        if z + u_h > H:
            if current_items:
                pallets.append({"items": current_items, "weight": current_weight})
            current_items = []
            current_weight = 0.0
            x, y, z = 0.0, 0.0, 0.0
            current_shelf_height, current_shelf_y = 0.0, 0.0

        # GRAVITY CHECK: Verify support ratio before placement
        support_ratio = compute_support_ratio(
            candidate_x=x,
            candidate_y=y,
            candidate_z=z,
            candidate_l=u_l,
            candidate_w=u_w,
            placed_items=current_items,
            debug=debug_support
        )
        
        # DEBUG LOG: Support check applied
        if debug_support and z > 1e-6:
            print(f"  [SUPPORT CHECK] box={urun.urun_kodu}, z={z:.2f}, "
                  f"support={support_ratio:.2%}")
        
        # FORCE NEW PALLET if insufficient support (shelf-based can't easily reposition)
        if support_ratio < min_support_ratio:
            if debug_support:
                print(f"  [SUPPORT REJECT] box={urun.urun_kodu}, z={z:.2f}, "
                      f"support={support_ratio:.2%}, required={min_support_ratio:.2%} "
                      f"(shelf-based forcing new pallet)")
            if current_items:
                pallets.append({"items": current_items, "weight": current_weight})
            current_items = []
            current_weight = 0.0
            x, y, z = 0.0, 0.0, 0.0
            current_shelf_height, current_shelf_y = 0.0, 0.0

        current_items.append({
            "urun": urun,
            "x": x, "y": y, "z": z,
            "L": u_l, "W": u_w, "H": u_h
        })
        current_weight += u_wgt
        
        x += u_l
        if u_h > current_shelf_height:
            current_shelf_height = u_h
        if u_w > current_shelf_y:
            current_shelf_y = u_w
        
    if current_items:
        pallets.append({"items": current_items, "weight": current_weight})
        
    return pallets


def basit_palet_paketleme(chromosome, palet_cfg, min_support_ratio=0.40, debug_support=False):
    """
    Kromozomdan paletleri oluşturur (with gravity constraint).
    
    Args:
        chromosome: (urunler, rotations) tuple
        palet_cfg: PaletConfig nesnesi
        min_support_ratio: Minimum support ratio (default 0.40)
        debug_support: Enable debug logging (default False)
        
    Returns:
        list[dict]: Her palet için placements ve weight
    """
    urunler, rotations = chromosome
    # CRITICAL: Pass gravity constraint parameters to pack_shelf_based
    pallets = pack_shelf_based(urunler, rotations, palet_cfg, min_support_ratio, debug_support)
    
    result = []
    for pallet in pallets:
        placements = []
        for item in pallet['items']:
            placements.append({
                'urun': item['urun'],
                'x': item['x'],
                'y': item['y'],
                'z': item['z'],
                'L': item['L'],
                'W': item['W'],
                'H': item['H']
            })
        result.append({
            'placements': placements,
            'weight': pallet['weight']
        })
    
    return result
