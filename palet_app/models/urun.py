from django.db import models

class Urun(models.Model):
    """Ürün modeli, ürün bilgilerini ve özelliklerini saklar."""
    
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
    
    def hacim(self):
        """Ürünün hacmini cm³ cinsinden hesaplar."""
        return self.boy * self.en * self.yukseklik
    
    def taban_alani(self):
        """Ürünün kapladığı taban alanını cm² cinsinden hesaplar."""
        return self.boy * self.en
    
    def boyutlar(self):
        """Ürünün boyutlarını (boy, en, yükseklik) döndürür."""
        return (self.boy, self.en, self.yukseklik)
    
    def possible_orientations(self):
        """Ürünün olası tüm yönelimlerini döndürür."""
        if not self.donus_serbest:
            return [(self.boy, self.en, self.yukseklik)]
        else:
            # Döndürülebilir ürünler için 6 farklı yönelim
            return [
                (self.boy, self.en, self.yukseklik),
                (self.boy, self.yukseklik, self.en),
                (self.en, self.boy, self.yukseklik),
                (self.en, self.yukseklik, self.boy),
                (self.yukseklik, self.boy, self.en),
                (self.yukseklik, self.en, self.boy)
            ] 