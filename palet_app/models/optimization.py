from django.db import models
import json
from django.utils import timezone


class Optimization(models.Model):
    """Optimizasyon islemi; container bilgisi JSON'dan (container_*) gelir."""

    palet_tipi = models.PositiveSmallIntegerField(null=True, blank=True, verbose_name="Palet Tipi (kullanilmiyor)")

    container_length = models.FloatField(null=True, blank=True, verbose_name="Container Uzunluk (cm)")
    container_width = models.FloatField(null=True, blank=True, verbose_name="Container Genişlik (cm)")
    container_height = models.FloatField(null=True, blank=True, verbose_name="Container Yükseklik (cm)")
    container_weight = models.FloatField(null=True, blank=True, verbose_name="Container Max Ağırlık (kg)")

    baslangic_zamani = models.DateTimeField(auto_now_add=True, verbose_name="Başlangıç Zamanı")
    bitis_zamani = models.DateTimeField(null=True, blank=True, verbose_name="Bitiş Zamanı")
    tamamlandi = models.BooleanField(default=False, verbose_name="Tamamlandı mı?")

    toplam_palet = models.PositiveIntegerField(default=0, verbose_name="Toplam Palet Sayısı")
    single_palet = models.PositiveIntegerField(default=0, verbose_name="Single Palet Sayısı")
    mix_palet = models.PositiveIntegerField(default=0, verbose_name="Mix Palet Sayısı")

    islem_durumu = models.TextField(default='{"current_step": 0, "total_steps": 5, "messages": []}', verbose_name="İşlem Durumu")
    yerlesmemis_urunler = models.JSONField(default=list, verbose_name="Yerleştirilemeyen Ürünler")
    algoritma = models.CharField(max_length=20, default='greedy', verbose_name="Algoritma")

    class Meta:
        verbose_name = "Optimizasyon"
        verbose_name_plural = "Optimizasyonlar"
        ordering = ['-baslangic_zamani']

    def __str__(self):
        return f"Optimizasyon #{self.id} ({self.algoritma})"

    def tamamla(self):
        self.bitis_zamani = timezone.now()
        self.tamamlandi = True
        self.save()

    def islem_adimi_ekle(self, mesaj):
        durum = json.loads(self.islem_durumu)
        durum['messages'].append(mesaj)
        durum['current_step'] += 1
        total = max(1, durum.get('total_steps', 5))
        durum['current_step'] = min(durum['current_step'], total)
        self.islem_durumu = json.dumps(durum)
        self.save()

    def get_islem_durumu(self):
        return json.loads(self.islem_durumu)
