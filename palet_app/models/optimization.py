from django.db import models
import json
from django.utils import timezone

class Optimization(models.Model):
    """
    Optimizasyon işlemi bilgilerini saklar. 
    Her optimizasyon işlemi birden fazla palet içerebilir.
    """
    
    PALET_TIPI_CHOICES = [
        (1, "Avrupa Paleti (120x80x180 cm)"),
        (2, "ISO Standart (120x100x200 cm)"),
        (3, "Asya Paleti (110x110x190 cm)"),
        (4, "ABD Paleti (121.9x101.6x200 cm)")
    ]
    
    palet_tipi = models.PositiveSmallIntegerField(choices=PALET_TIPI_CHOICES, null=True, blank=True, verbose_name="Palet Tipi")
    
    # Dinamik container bilgileri (JSON'dan gelen)
    container_length = models.FloatField(null=True, blank=True, verbose_name="Container Uzunluk (cm)")
    container_width = models.FloatField(null=True, blank=True, verbose_name="Container Genişlik (cm)")
    container_height = models.FloatField(null=True, blank=True, verbose_name="Container Yükseklik (cm)")
    container_weight = models.FloatField(null=True, blank=True, verbose_name="Container Max Ağırlık (kg)")
    
    baslangic_zamani = models.DateTimeField(auto_now_add=True, verbose_name="Başlangıç Zamanı")
    bitis_zamani = models.DateTimeField(null=True, blank=True, verbose_name="Bitiş Zamanı")
    tamamlandi = models.BooleanField(default=False, verbose_name="Tamamlandı mı?")
    
    # İşlem istatistikleri
    toplam_palet = models.PositiveIntegerField(default=0, verbose_name="Toplam Palet Sayısı")
    single_palet = models.PositiveIntegerField(default=0, verbose_name="Single Palet Sayısı")
    mix_palet = models.PositiveIntegerField(default=0, verbose_name="Mix Palet Sayısı")
    
    # Grafikler için kullanılacak dosyalar
    pie_chart = models.ImageField(upload_to='optimizasyon_grafikler/', null=True, blank=True, verbose_name="Pasta Grafiği")
    bar_chart = models.ImageField(upload_to='optimizasyon_grafikler/', null=True, blank=True, verbose_name="Çubuk Grafiği")
    
    # İşlem durumu bilgileri - ön yüzde göstermek için
    islem_durumu = models.TextField(default='{"current_step": 0, "total_steps": 5, "messages": []}', verbose_name="İşlem Durumu")
    
    # Yerleştirilemeyen ürünler
    yerlesmemis_urunler = models.JSONField(default=list, verbose_name="Yerleştirilemeyen Ürünler")
    
    # Kullanılan algoritma
    algoritma = models.CharField(max_length=20, default='greedy', verbose_name="Algoritma")
    
    class Meta:
        verbose_name = "Optimizasyon"
        verbose_name_plural = "Optimizasyonlar"
        ordering = ['-baslangic_zamani']
    
    def __str__(self):
        return f"Optimizasyon #{self.id} - {self.get_palet_tipi_display()}"
    
    def tamamla(self):
        """Optimizasyon işlemini tamamlar."""
        self.bitis_zamani = timezone.now()
        self.tamamlandi = True
        self.save()
    
    def islem_adimi_ekle(self, mesaj):
        """İşlem adımlarına yeni bir mesaj ekler."""
        durum = json.loads(self.islem_durumu)
        durum['messages'].append(mesaj)
        durum['current_step'] += 1
        self.islem_durumu = json.dumps(durum)
        self.save()
    
    def get_islem_durumu(self):
        """İşlem durumunu dict olarak döner."""
        return json.loads(self.islem_durumu) 