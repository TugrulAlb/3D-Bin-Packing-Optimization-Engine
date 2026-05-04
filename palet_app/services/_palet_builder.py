"""Ortak Palet builder."""

from ..models import Palet


def create_django_palet(placements, palet_cfg, optimization, palet_id, palet_turu):
    """Yerleşim listesinden Palet kaydı oluşturur.

    placements item formatı: ``{'urun': Urun, 'x', 'y', 'z', 'L', 'W', 'H'}``.
    """
    palet = Palet(
        optimization=optimization,
        palet_id=palet_id,
        palet_tipi=None,
        palet_turu=palet_turu,
        custom_en=palet_cfg.width,
        custom_boy=palet_cfg.length,
        custom_max_yukseklik=palet_cfg.height,
        custom_max_agirlik=palet_cfg.max_weight,
    )

    urun_konumlari = {}
    urun_boyutlari = {}
    toplam_agirlik = 0.0
    kullanilan_hacim = 0.0

    for item in placements:
        urun = item['urun']
        urun_id = str(urun.id)
        urun_konumlari[urun_id] = [item['x'], item['y'], item['z']]
        urun_boyutlari[urun_id] = [item['L'], item['W'], item['H']]
        toplam_agirlik += urun.agirlik
        kullanilan_hacim += (item['L'] * item['W'] * item['H'])

    palet.urun_konumlari = urun_konumlari
    palet.urun_boyutlari = urun_boyutlari
    palet.toplam_agirlik = toplam_agirlik
    palet.kullanilan_hacim = kullanilan_hacim
    palet.save()

    return palet
