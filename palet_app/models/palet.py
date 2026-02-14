from django.db import models
import json
from .optimization import Optimization

class Palet(models.Model):
    """Palet modeli, palet özellikleri ve içindeki ürünleri saklar."""
    
    # Palet tipleri ve özellikleri
    PALET_TIPLERI = {
        1: {"ad": "Avrupa Paleti", "en": 120, "boy": 80, "max_yukseklik": 180, "max_agirlik": 15000},
        2: {"ad": "ISO Standart", "en": 120, "boy": 100, "max_yukseklik": 200, "max_agirlik": 20000},
        3: {"ad": "Asya Paleti", "en": 110, "boy": 110, "max_yukseklik": 190, "max_agirlik": 1800},
        4: {"ad": "ABD Paleti", "en": 121.9, "boy": 101.6, "max_yukseklik": 200, "max_agirlik": 2500}
    }
    
    PALET_TIPI_CHOICES = [
        (1, "Avrupa Paleti (120x80x180 cm)"),
        (2, "ISO Standart (120x100x200 cm)"),
        (3, "Asya Paleti (110x110x190 cm)"),
        (4, "ABD Paleti (121.9x101.6x200 cm)")
    ]
    
    PALET_TURU_CHOICES = [
        ('single', 'Single Palet'),
        ('mix', 'Mix Palet')
    ]
    
    optimization = models.ForeignKey(Optimization, on_delete=models.CASCADE, related_name='paletler', verbose_name="Optimizasyon")
    palet_id = models.PositiveIntegerField(verbose_name="Palet ID")
    palet_tipi = models.PositiveSmallIntegerField(choices=PALET_TIPI_CHOICES, null=True, blank=True, verbose_name="Palet Tipi")
    palet_turu = models.CharField(max_length=10, choices=PALET_TURU_CHOICES, default='mix', verbose_name="Palet Türü")
    
    # Dinamik palet boyutları (JSON'dan gelen container bilgisi için)
    custom_en = models.FloatField(null=True, blank=True, verbose_name="Özel En (cm)")
    custom_boy = models.FloatField(null=True, blank=True, verbose_name="Özel Boy (cm)")
    custom_max_yukseklik = models.FloatField(null=True, blank=True, verbose_name="Özel Max Yükseklik (cm)")
    custom_max_agirlik = models.FloatField(null=True, blank=True, verbose_name="Özel Max Ağırlık (kg)")
    
    # Paletin doluluk bilgileri
    toplam_agirlik = models.FloatField(default=0.0, verbose_name="Toplam Ağırlık (kg)")
    kullanilan_hacim = models.FloatField(default=0.0, verbose_name="Kullanılan Hacim (cm³)")
    
    # Görselleştirme için gerekli alanlar
    gorsel = models.ImageField(upload_to='palet_gorseller/', null=True, blank=True, verbose_name="Palet Görseli")
    urun_konumlari = models.JSONField(default=dict, verbose_name="Ürün Konumları")
    urun_boyutlari = models.JSONField(default=dict, verbose_name="Ürün Boyutları")
    
    class Meta:
        verbose_name = "Palet"
        verbose_name_plural = "Paletler"
        ordering = ['palet_id']
        unique_together = [['optimization', 'palet_id']]
    
    def __str__(self):
        return f"Palet {self.palet_id} ({self.get_palet_tipi_display()}, {self.palet_turu})"
    
    @property
    def en(self):
        """Paletin enini döndürür (custom veya standart)"""
        if self.custom_en is not None:
            return self.custom_en
        return self.PALET_TIPLERI[self.palet_tipi]["en"] if self.palet_tipi else 100
    
    @property
    def boy(self):
        """Paletin boyunu döndürür (custom veya standart)"""
        if self.custom_boy is not None:
            return self.custom_boy
        return self.PALET_TIPLERI[self.palet_tipi]["boy"] if self.palet_tipi else 120
    
    @property
    def max_yukseklik(self):
        """Paletin max yüksekliğini döndürür (custom veya standart)"""
        if self.custom_max_yukseklik is not None:
            return self.custom_max_yukseklik
        return self.PALET_TIPLERI[self.palet_tipi]["max_yukseklik"] if self.palet_tipi else 180
    
    @property
    def max_agirlik(self):
        """Paletin max ağırlığını döndürür (custom veya standart)"""
        if self.custom_max_agirlik is not None:
            return self.custom_max_agirlik
        return self.PALET_TIPLERI[self.palet_tipi]["max_agirlik"] if self.palet_tipi else 1250
    
    def hacim(self):
        """Paletin toplam hacmini cm³ cinsinden hesaplar."""
        return self.en * self.boy * self.max_yukseklik
    
    def taban_alani(self):
        """Paletin taban alanını cm² cinsinden hesaplar."""
        return self.en * self.boy
    
    def doluluk_orani(self):
        """Paletin doluluk oranını yüzde olarak hesaplar."""
        return (self.kullanilan_hacim / self.hacim()) * 100
    
    def json_to_dict(self, json_string):
        """JSON formatındaki stringi dict'e çevirir"""
        if isinstance(json_string, str):
            return json.loads(json_string)
        return json_string 