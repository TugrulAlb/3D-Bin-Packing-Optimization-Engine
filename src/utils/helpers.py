"""
Yardımcı Fonksiyonlar
======================

Algoritma genelinde kullanılan yardımcı araçlar.
"""


def urun_hacmi(urun) -> float:
    """Ürün hacmini hesaplar (cm³)."""
    return urun.boy * urun.en * urun.yukseklik


def urun_agirlik(urun) -> float:
    """Ürün ağırlığını döndürür (kg)."""
    return urun.agirlik


def possible_orientations_for(urun):
    """
    Ürünün olası yatay düzlem yönelimlerini döndürür.
    
    Yükseklik sabittir (Z ekseni), sadece X-Y düzleminde
    90° dönüş yapılabilir.
    
    Args:
        urun: UrunData nesnesi
        
    Returns:
        list[tuple]: (boy, en, yükseklik) tuple listesi
    """
    if not urun.donus_serbest:
        return [(urun.boy, urun.en, urun.yukseklik)]
    return [
        (urun.boy, urun.en, urun.yukseklik),
        (urun.en, urun.boy, urun.yukseklik),
    ]


def group_products_smart(urunler):
    """
    Ürünleri Code + Boyut + Ağırlık kombinasyonuna göre gruplar.
    
    Args:
        urunler: UrunData listesi
        
    Returns:
        dict: (code, boy, en, yükseklik, ağırlık) → [UrunData] mapping
    """
    groups = {}
    for u in urunler:
        key = (u.urun_kodu, u.boy, u.en, u.yukseklik, u.agirlik)
        if key not in groups:
            groups[key] = []
        groups[key].append(u)
    return groups
