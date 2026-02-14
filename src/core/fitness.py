"""
Fitness Değerlendirme Modülü
==============================

Genetik Algoritma bireylerinin (kromozomların) kalitesini ölçer.

Değerlendirme Kriterleri (Öncelik Sırasıyla):
    1. Palet sayısını minimize et (EN YÜKSEK ÖNCELİK)
    2. Doluluk oranını maksimize et
    3. Fiziksel kısıtları asla ihlal etme (KIRMIZI ÇİZGİ)
    4. Ağırlık merkezi dengesini koru
    5. İstifleme kurallarına uy
"""

from dataclasses import dataclass
from .packing import pack_maximal_rectangles
from ..utils.helpers import urun_hacmi
from ..models.container import PaletConfig


# ====================================================================
# ADAPTİF AĞIRLIK SİSTEMİ
# ====================================================================

class AdaptiveWeights:
    """
    Performansa göre otomatik ayarlanan fitness ağırlıkları.
    
    GA ilerledikçe mevcut çözüm kalitesine göre ağırlıkları
    dinamik olarak ayarlar.
    """
    
    def __init__(self):
        # Başlangıç değerleri
        self.w_pallet_count = 15000
        self.w_optimal_bonus = 150000
        self.w_volume = 20000
        self.w_weight_violation = 1000000      # Sabit (kırmızı çizgi)
        self.w_physical_violation = 10000000    # Sabit (kırmızı çizgi)
        self.w_cog_penalty = 0                  # Devre dışı
        self.w_stacking_penalty = 100000        # Sabit
        
        # HARD CAPS
        self.MAX_VOLUME = 40000
        self.MAX_PALLET_COUNT = 50000
        
    def adapt(self, best_chromosome, theo_min_pallets):
        """Performansa göre ağırlıkları ayarla."""
        if not best_chromosome:
            return
            
        palet_sayisi = best_chromosome.palet_sayisi
        doluluk = best_chromosome.ortalama_doluluk
        
        # KURAL 1: Palet sayısı fazlaysa → pallet_count artır
        if palet_sayisi > theo_min_pallets + 2:
            self.w_pallet_count = min(self.w_pallet_count * 1.1, self.MAX_PALLET_COUNT)
            print(f"  🔧 Palet sayısı yüksek → w_pallet_count artırıldı: {self.w_pallet_count:.0f}")
        elif palet_sayisi <= theo_min_pallets:
            self.w_pallet_count = max(self.w_pallet_count * 0.95, 10000)
            
        # KURAL 2: Doluluk düşükse → volume weight artır
        if doluluk < 0.65:
            self.w_volume = min(self.w_volume * 1.1, self.MAX_VOLUME)
            print(f"  🔧 Doluluk düşük (%{doluluk*100:.1f}) → w_volume artırıldı: {self.w_volume:.0f}")
        elif doluluk > 0.85:
            self.w_volume = max(self.w_volume * 0.98, 15000)
            
        # KURAL 3: Optimal bonus'u dinamik ayarla
        if palet_sayisi == theo_min_pallets:
            self.w_optimal_bonus = min(self.w_optimal_bonus * 1.1, 300000)
        else:
            self.w_optimal_bonus = max(self.w_optimal_bonus * 0.95, 100000)
    
    def to_dict(self):
        return {
            "w_pallet_count": self.w_pallet_count,
            "w_optimal_bonus": self.w_optimal_bonus,
            "w_volume": self.w_volume,
            "w_weight_violation": self.w_weight_violation,
            "w_physical_violation": self.w_physical_violation,
            "w_cog_penalty": self.w_cog_penalty,
            "w_stacking_penalty": self.w_stacking_penalty
        }


# Global adaptive weights instance
_adaptive_weights = AdaptiveWeights()


def get_weights():
    """Mevcut ağırlıkları döndür."""
    return _adaptive_weights.to_dict()


def adapt_weights(best_chromosome, theo_min_pallets):
    """Ağırlıkları performansa göre ayarla."""
    _adaptive_weights.adapt(best_chromosome, theo_min_pallets)


# Geriye uyumluluk
GA_WEIGHTS = _adaptive_weights.to_dict()


# ====================================================================
# FITNESS SONUÇ YAPISI
# ====================================================================

@dataclass
class FitnessResult:
    """Fitness hesaplama sonucu."""
    fitness: float
    palet_sayisi: int
    ortalama_doluluk: float


# ====================================================================
# YARDIMCI FONKSİYONLAR
# ====================================================================

def calculate_center_of_gravity(items):
    """Paletin ağırlık merkezini hesaplar."""
    if not items:
        return 0, 0, 0
    
    total_weight = 0
    weighted_x = 0
    weighted_y = 0
    weighted_z = 0
    
    for item in items:
        center_x = item['x'] + item['L'] / 2
        center_y = item['y'] + item['W'] / 2
        center_z = item['z'] + item['H'] / 2
        
        weight = item['urun'].agirlik
        
        weighted_x += center_x * weight
        weighted_y += center_y * weight
        weighted_z += center_z * weight
        total_weight += weight
    
    if total_weight == 0:
        return 0, 0, 0
    
    cog_x = weighted_x / total_weight
    cog_y = weighted_y / total_weight
    cog_z = weighted_z / total_weight
    
    return cog_x, cog_y, cog_z


def check_stacking_violations(items):
    """
    İstifleme hatalarını kontrol eder (havada duran kutular).
    Minimum %70 destek alanı gerektirir.
    """
    violations = 0
    
    for i, item in enumerate(items):
        if item['z'] == 0:
            continue
        
        item_area = item['L'] * item['W']
        supported_area = 0.0
        
        item_bottom = item['z']
        item_x1 = item['x']
        item_x2 = item['x'] + item['L']
        item_y1 = item['y']
        item_y2 = item['y'] + item['W']
        
        for j, other in enumerate(items):
            if i == j:
                continue
            
            other_top = other['z'] + other['H']
            
            if abs(other_top - item_bottom) < 0.1:
                other_x1 = other['x']
                other_x2 = other['x'] + other['L']
                other_y1 = other['y']
                other_y2 = other['y'] + other['W']
                
                overlap_x = max(0, min(item_x2, other_x2) - max(item_x1, other_x1))
                overlap_y = max(0, min(item_y2, other_y2) - max(item_y1, other_y1))
                overlap_area = overlap_x * overlap_y
                
                supported_area += overlap_area
        
        support_ratio = supported_area / item_area if item_area > 0 else 0
        if support_ratio < 0.70:
            violations += 1
    
    return violations


# ====================================================================
# ANA FITNESS FONKSİYONU
# ====================================================================

def evaluate_fitness(chromosome, palet_cfg: PaletConfig) -> FitnessResult:
    """
    Kromozomun başarısını ölçer - Adaptif Ağırlıklar ile.
    
    Motor AUTO-ORIENTATION kullanır (rot_gen gerekmez).
    GA yalnızca SEQUENCE (ürün sırası) optimize eder.
    """
    weights = get_weights()
    
    # 1. Yerleştirme Motorunu Çalıştır (Maximal Rectangles + Auto-Orientation)
    pallets = pack_maximal_rectangles(chromosome.urunler, palet_cfg)
    
    if not pallets:
        chromosome.fitness = -1e9
        return FitnessResult(-1e9, 0, 0.0)

    P_GA = len(pallets)
    
    # 2. Teorik Minimum Palet Sayısı
    total_load_vol = sum(urun_hacmi(u) for u in chromosome.urunler)
    theo_min = max(1, int(total_load_vol / palet_cfg.volume) + 1)
    
    # 3. FITNESS HESAPLAMA
    fitness_score = 0.0
    total_fill_ratio = 0.0
    has_violation = False
    
    # --- ÖNCELİK 1: PALET SAYISI ---
    if P_GA == theo_min:
        fitness_score += weights["w_optimal_bonus"]
    elif P_GA < theo_min:
        fitness_score += weights["w_optimal_bonus"] * 2
    else:
        extra_pallets = P_GA - theo_min
        fitness_score -= weights["w_pallet_count"] * extra_pallets
    
    # --- ÖNCELİK 2: DOLULUK ORANI ---
    for pallet in pallets:
        p_vol = sum(i["L"] * i["W"] * i["H"] for i in pallet["items"])
        fill_ratio = p_vol / palet_cfg.volume
        total_fill_ratio += fill_ratio
        fitness_score += weights["w_volume"] * (fill_ratio ** 4)
    
    avg_doluluk = total_fill_ratio / P_GA if P_GA > 0 else 0.0
    
    # --- KIRMIZI ÇİZGİ: FİZİKSEL KISIT İHLALLERİ ---
    for pallet in pallets:
        if pallet["weight"] > palet_cfg.max_weight:
            fitness_score -= weights["w_weight_violation"]
            has_violation = True
        
        for item in pallet["items"]:
            if (item["x"] + item["L"] > palet_cfg.length or 
                item["y"] + item["W"] > palet_cfg.width or 
                item["z"] + item["H"] > palet_cfg.height):
                fitness_score -= weights["w_physical_violation"]
                has_violation = True
        
        if weights["w_cog_penalty"] > 0:
            cog_x, cog_y, cog_z = calculate_center_of_gravity(pallet["items"])
            palet_center_x = palet_cfg.length / 2
            palet_center_y = palet_cfg.width / 2
            distance = ((cog_x - palet_center_x)**2 + (cog_y - palet_center_y)**2)**0.5
            if distance > 10:
                penalty = int((distance - 10) / 10)
                fitness_score -= weights["w_cog_penalty"] * penalty
        
        stacking_violations = check_stacking_violations(pallet["items"])
        if stacking_violations > 0:
            fitness_score -= weights["w_stacking_penalty"] * stacking_violations
            has_violation = True
    
    if has_violation:
        fitness_score = min(fitness_score, -1e6)
    
    # Sonuç
    chromosome.fitness = fitness_score
    chromosome.palet_sayisi = P_GA
    chromosome.ortalama_doluluk = avg_doluluk
    
    return FitnessResult(fitness_score, P_GA, avg_doluluk)
