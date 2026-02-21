from django.db import models


class Urun(models.Model):
    """Urun modeli; optimizasyon sirasinda DB'ye yazilir, algo src.utils.helpers ile hesaplar."""

    urun_kodu = models.CharField(max_length=20, verbose_name="Ürün Kodu")
    urun_adi = models.CharField(max_length=100, verbose_name="Ürün Adı")
    boy = models.FloatField(verbose_name="Boy (cm)")
    en = models.FloatField(verbose_name="En (cm)")
    yukseklik = models.FloatField(verbose_name="Yükseklik (cm)")
    agirlik = models.FloatField(verbose_name="Ağırlık (kg)")
    mukavemet = models.FloatField(verbose_name="Mukavemet (kg)")
    donus_serbest = models.BooleanField(default=False, verbose_name="Dönüş Serbest")
    istiflenebilir = models.BooleanField(default=True, verbose_name="İstiflenebilir")

    class Meta:
        verbose_name = "Ürün"
        verbose_name_plural = "Ürünler"

    def __str__(self):
        return f"{self.urun_adi} ({self.urun_kodu})"
