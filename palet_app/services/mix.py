"""Mix palet servisleri — kromozomdan ve packer dict'ten Django'ya."""

from src.core.packing import compact_pallet, merge_and_repack
from src.core.packing_first_fit import pack_maximal_rectangles_first_fit

from ..models import Palet
from ._palet_builder import create_django_palet


def chromosome_to_palets(chromosome, palet_cfg, optimization, baslangic_id):
    """
    En iyi kromozomdan Django Palet nesneleri oluşturur.

    Adımlar:
        1. Kromozom decode → ürün sırası
        2. pack_maximal_rectangles_first_fit → ham palet dict listesi
        3. compact_pallet → gravity + origin sıkıştırma
        4. merge_and_repack → düşük doluluklu paletleri birleştir
        5. Django Palet nesnelerine dönüştür ve kaydet
    """
    siralanmis_urunler = [chromosome.urunler[i] for i in chromosome.sira_gen]
    pallets = pack_maximal_rectangles_first_fit(siralanmis_urunler, palet_cfg)

    for p in pallets:
        compact_pallet(p, palet_cfg)

    onceki_sayi = len(pallets)
    pallets = merge_and_repack(pallets, palet_cfg)
    sonraki_sayi = len(pallets)
    if sonraki_sayi < onceki_sayi:
        print(
            f"[MERGE & REPACK] {onceki_sayi} → {sonraki_sayi} palet "
            f"({onceki_sayi - sonraki_sayi} palet azaltıldı)"
        )

    django_paletler = []
    palet_id = baslangic_id

    for pallet_data in pallets:
        palet = create_django_palet(
            pallet_data['items'], palet_cfg, optimization, palet_id, 'mix',
            items_are_dicts=True,
        )
        django_paletler.append(palet)
        palet_id += 1

    return django_paletler


def mix_palet_data_to_django(mix_palet_data, palet_cfg, optimization):
    """
    Greedy mix çıktısı dict listesini Django Palet nesnelerine dönüştürür.
    """
    django_paletler = []

    for pallet_dict in mix_palet_data:
        palet = Palet(
            optimization=optimization,
            palet_id=pallet_dict['id'],
            palet_tipi=None,
            palet_turu='mix',
            custom_en=palet_cfg.width,
            custom_boy=palet_cfg.length,
            custom_max_yukseklik=palet_cfg.height,
            custom_max_agirlik=palet_cfg.max_weight,
        )

        urun_konumlari = {}
        urun_boyutlari = {}
        toplam_agirlik = 0.0
        kullanilan_hacim = 0.0

        for item in pallet_dict.get('items', []):
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

        django_paletler.append(palet)

    return django_paletler
