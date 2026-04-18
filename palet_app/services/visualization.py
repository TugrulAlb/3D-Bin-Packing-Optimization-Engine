"""Görselleştirme servisleri — Matplotlib 3D + Plotly özet grafikler."""

from django.core.files.base import ContentFile

from src.utils.visualization import render_pallet_3d, render_summary_charts

from ..models import Palet


def palet_gorsellestir(palet, urunler, save_to_file=True):
    """
    Matplotlib ile 3D palet görselleştirme (Django entegrasyonu).

    Args:
        palet: Django Palet modeli
        urunler: Django Urun queryset
        save_to_file: True ise ContentFile döndürür

    Returns:
        ContentFile veya BytesIO
    """
    PL, PW, PH = palet.boy, palet.en, palet.max_yukseklik

    urun_konumlari = palet.json_to_dict(palet.urun_konumlari)
    urun_boyutlari = palet.json_to_dict(palet.urun_boyutlari)

    items = []
    for urun in urunler:
        uid = str(urun.id)
        if uid not in urun_konumlari:
            continue

        pos = urun_konumlari[uid]
        dim = urun_boyutlari[uid]

        if isinstance(pos, list):
            pos = tuple(pos)
        if isinstance(dim, list):
            dim = tuple(dim)

        items.append({
            'urun_kodu': urun.urun_kodu,
            'x': pos[0], 'y': pos[1], 'z': pos[2],
            'L': dim[0], 'W': dim[1], 'H': dim[2],
        })

    title = (
        f'Palet {palet.palet_id} - {palet.palet_turu.upper()}\n'
        f'Doluluk: {palet.doluluk_orani():.1f}%'
    )

    buf = render_pallet_3d(PL, PW, PH, items, title=title)

    if save_to_file:
        return ContentFile(buf.read())
    return buf


def ozet_grafikler_olustur(optimization):
    """Özet grafikler oluşturur - Plotly ile interaktif HTML."""
    paletler = Palet.objects.filter(optimization=optimization)
    single = paletler.filter(palet_turu='single').count()
    mix = paletler.filter(palet_turu='mix').count()

    optimization.single_palet = single
    optimization.mix_palet = mix
    optimization.toplam_palet = single + mix
    optimization.save()

    palet_data_list = []
    for p in paletler:
        palet_data_list.append({
            'palet_id': p.palet_id,
            'palet_turu': p.palet_turu,
            'doluluk': p.doluluk_orani(),
        })

    return render_summary_charts(palet_data_list)
