from django.db import models
import json
from .optimization import Optimization


class Palet(models.Model):
    """Palet modeli; boyutlar JSON container bilgisinden (custom_*) gelir."""

    PALET_TURU_CHOICES = [
        ('single', 'Single Palet'),
        ('mix', 'Mix Palet')
    ]

    optimization = models.ForeignKey(Optimization, on_delete=models.CASCADE, related_name='paletler', verbose_name="Optimizasyon")
    palet_id = models.PositiveIntegerField(verbose_name="Palet ID")
    palet_tipi = models.PositiveSmallIntegerField(null=True, blank=True, verbose_name="Palet Tipi (kullanilmiyor, custom kullanilir)")
    palet_turu = models.CharField(max_length=10, choices=PALET_TURU_CHOICES, default='mix', verbose_name="Palet Türü")

    # Dinamik palet boyutlari (JSON'dan gelen container bilgisi)
    custom_en = models.FloatField(null=True, blank=True, verbose_name="Özel En (cm)")
    custom_boy = models.FloatField(null=True, blank=True, verbose_name="Özel Boy (cm)")
    custom_max_yukseklik = models.FloatField(null=True, blank=True, verbose_name="Özel Max Yükseklik (cm)")
    custom_max_agirlik = models.FloatField(null=True, blank=True, verbose_name="Özel Max Ağırlık (kg)")

    toplam_agirlik = models.FloatField(default=0.0, verbose_name="Toplam Ağırlık (kg)")
    kullanilan_hacim = models.FloatField(default=0.0, verbose_name="Kullanılan Hacim (cm³)")

    gorsel = models.ImageField(upload_to='palet_gorseller/', null=True, blank=True, verbose_name="Palet Görseli")
    urun_konumlari = models.JSONField(default=dict, verbose_name="Ürün Konumları")
    urun_boyutlari = models.JSONField(default=dict, verbose_name="Ürün Boyutları")

    class Meta:
        verbose_name = "Palet"
        verbose_name_plural = "Paletler"
        ordering = ['palet_id']
        unique_together = [['optimization', 'palet_id']]

    def __str__(self):
        return f"Palet {self.palet_id} ({self.palet_turu})"

    @property
    def en(self):
        if self.custom_en is not None:
            return self.custom_en
        return 100.0

    @property
    def boy(self):
        if self.custom_boy is not None:
            return self.custom_boy
        return 120.0

    @property
    def max_yukseklik(self):
        if self.custom_max_yukseklik is not None:
            return self.custom_max_yukseklik
        return 180.0

    @property
    def max_agirlik(self):
        if self.custom_max_agirlik is not None:
            return self.custom_max_agirlik
        return 1250.0

    def hacim(self):
        """doluluk_orani() tarafindan kullanilir."""
        return self.en * self.boy * self.max_yukseklik

    def doluluk_orani(self):
        return (self.kullanilan_hacim / self.hacim()) * 100

    def json_to_dict(self, json_string):
        if isinstance(json_string, str):
            return json.loads(json_string)
        return json_string
