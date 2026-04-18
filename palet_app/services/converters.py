"""Django Model ↔ Algoritma nesnesi dönüştürücüleri."""

from src.models import PaletConfig, UrunData


def django_urun_to_urundata(urun):
    """Django Urun modelini UrunData nesnesine çevirir."""
    urun_data = UrunData(
        urun_id=urun.id,
        code=urun.urun_kodu,
        boy=urun.boy,
        en=urun.en,
        yukseklik=urun.yukseklik,
        agirlik=urun.agirlik,
        quantity=1,
        is_package=False,
    )
    urun_data.donus_serbest = urun.donus_serbest
    urun_data.mukavemet = urun.mukavemet
    return urun_data


def container_info_to_config(container_info):
    """Container bilgisini PaletConfig nesnesine çevirir."""
    return PaletConfig(
        length=container_info['length'],
        width=container_info['width'],
        height=container_info['height'],
        max_weight=container_info['weight'],
    )
