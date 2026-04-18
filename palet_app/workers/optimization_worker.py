"""Arka plan optimizasyon worker'ı.

``run_optimization`` bir Django thread içinde çalışır; cancel checkpoint'leri
ile fazlar arası iptal sinyalini okur.
"""

import json
import time

from ..models import Urun, Optimization, Palet
from ..services import (
    single_palet_yerlestirme,
    chromosome_to_palets,
    mix_palet_data_to_django,
    merge_repack_service,
    merge_repack_mix_service,
    palet_gorsellestir,
)
from src.models import PaletConfig, UrunData
from src.core.packing_first_fit import pack_maximal_rectangles_first_fit

from .cancel_registry import OptimizationCancelled, check_cancel
from .progress import estimate_mix_sec


def run_greedy_mix(urun_data_listesi, palet_cfg, start_id):
    """Saf greedy mix palet yerleştirme (First-Fit + Maximal Rectangles).

    mix_palet_data_to_django formatında sonuç döndürür.
    """
    pallets = pack_maximal_rectangles_first_fit(urun_data_listesi, palet_cfg)
    result = []
    pid = start_id
    for p in pallets:
        result.append({
            'id': pid,
            'type': 'MIX',
            'quantity': len(p['items']),
            'items': p['items'],
            'weight': p.get('weight', 0),
        })
        pid += 1
    return result


def run_optimization(urun_verileri, container_info, optimization_id, algoritma='greedy', ga_mode='balanced', group_id=None):
    """Arka planda çalışacak optimizasyon işlemi. Bir thread içinde çalışır.

    ga_mode: sadece genetic için kullanılır — 'fast', 'balanced', 'quality'.
    group_id: Benchmark grubu; sayfadan çıkılınca toplu iptal için.
    """
    try:
        check_cancel(optimization_id, group_id)
        optimization = Optimization.objects.get(id=optimization_id)
        # Tek doğru kaynak: başlatma anında kaydedilen algoritma (DB)
        algoritma = (optimization.algoritma or 'greedy').strip().lower()
        if algoritma not in ('genetic', 'differential_evolution', 'greedy'):
            algoritma = 'greedy'
        _thread_t0 = time.time()
        # Her algoritma kendi süresini t=0'dan başlasın (sıralı çalıştırmada
        # önceki algoritmaların beklemesi bu süreye yansımasın).
        from django.utils import timezone as _tz
        optimization.baslangic_zamani = _tz.now()
        optimization.save(update_fields=['baslangic_zamani'])
        print(f"\nrun_optimization() basladi (ID: {optimization_id}, Algoritma DB'den: {algoritma}, t0={_thread_t0:.3f})")
        print(f"Optimization objesi bulundu (ID: {optimization_id})")

        check_cancel(optimization_id, group_id)
        optimization.set_phase('baslangic')
        optimization.islem_adimi_ekle(f"[MOTOR] Algoritma: {algoritma.upper()}")

        # Adım 1: Ürünleri veritabanına kaydet
        check_cancel(optimization_id, group_id)
        optimization.set_phase('urunler')
        optimization.islem_adimi_ekle("Ürün verileri yükleniyor...")

        urunler = []
        for veri in urun_verileri:
            urun = Urun(
                urun_kodu=veri["urun_kodu"],
                urun_adi=veri["urun_adi"],
                boy=veri["boy"],
                en=veri["en"],
                yukseklik=veri["yukseklik"],
                agirlik=veri["agirlik"],
                mukavemet=veri.get("mukavemet", 100000),
                donus_serbest=veri.get("donus_serbest", True),
                istiflenebilir=veri.get("istiflenebilir", True),
            )
            urun.save()
            urunler.append(urun)

        # Adım 2: Single palet yerleştirme
        check_cancel(optimization_id, group_id)
        optimization.set_phase('single', expected_sec=max(3.0, len(urunler) * 0.04))
        optimization.islem_adimi_ekle("Single paletler oluşturuluyor...")
        single_paletler, yerlesmemis_urunler = single_palet_yerlestirme(urunler, container_info, optimization)

        palet_cfg = PaletConfig(
            length=container_info['length'],
            width=container_info['width'],
            height=container_info['height'],
            max_weight=container_info['weight'],
        )

        urun_data_listesi = []
        for urun in yerlesmemis_urunler:
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
            urun_data_listesi.append(urun_data)

        # Adım 3: Mix palet yerleştirme
        check_cancel(optimization_id, group_id)
        mix_motor_start = time.time()
        optimization.set_phase('mix', expected_sec=estimate_mix_sec(algoritma, len(urun_data_listesi)))
        optimization.islem_adimi_ekle(f"[TANI] Mix motoru başlıyor — yerleşmeyen ürün sayısı: {len(urun_data_listesi)}")

        if algoritma == 'genetic':
            from src.core.genetic_algorithm import run_ga, _adaptive_ga_params

            urun_sayisi = len(urun_data_listesi)
            ga_mode = (ga_mode or 'balanced').strip().lower()
            if ga_mode not in ('fast', 'balanced', 'quality'):
                ga_mode = 'balanced'

            base = _adaptive_ga_params(urun_sayisi)
            mode_scale = {
                'fast': {'pop': 0.7, 'gen': 0.6, 'mut': 0.25, 'k': 3},
                'balanced': {'pop': 1.0, 'gen': 1.0, 'mut': 0.20, 'k': 3},
                'quality': {'pop': 1.4, 'gen': 1.6, 'mut': 0.16, 'k': 4},
            }[ga_mode]

            pop_size = max(20, int(round(base['population_size'] * mode_scale['pop'])))
            generations = max(20, int(round(base['generations'] * mode_scale['gen'])))
            elitism = max(2, min(10, pop_size // 15))
            mutation_rate = mode_scale['mut']
            tournament_k = mode_scale['k']

            mode_label = {'fast': 'Hızlı', 'balanced': 'Dengeli', 'quality': 'Kaliteli'}[ga_mode]
            optimization.islem_adimi_ekle(
                f"Genetik Algoritma ({mode_label} mod) ile mix paletler olusturuluyor..."
            )
            optimization.islem_adimi_ekle(
                f"Mod: {ga_mode.upper()} | Pop={pop_size}, Nesil={generations}, "
                f"Elit={elitism}, Mut={mutation_rate}, K={tournament_k}, Ürün={urun_sayisi}"
            )

            best_chromosome, history = run_ga(
                urunler=urun_data_listesi,
                palet_cfg=palet_cfg,
                population_size=pop_size,
                generations=generations,
                mutation_rate=mutation_rate,
                tournament_k=tournament_k,
                elitism=elitism,
            )

            if best_chromosome:
                optimization.islem_adimi_ekle(
                    f"En iyi çözüm: Fitness={best_chromosome.fitness:.2f}, "
                    f"Palet={best_chromosome.palet_sayisi}, "
                    f"Doluluk={best_chromosome.ortalama_doluluk:.2%}"
                )
                mix_paletler = chromosome_to_palets(
                    best_chromosome,
                    palet_cfg,
                    optimization,
                    len(single_paletler) + 1,
                )
                optimization.islem_adimi_ekle(f"{len(mix_paletler)} adet mix palet oluşturuldu (GA).")
            else:
                optimization.islem_adimi_ekle("GA çözüm üretemedi, Greedy yönteme geçiliyor...")
                mix_palet_data = run_greedy_mix(urun_data_listesi, palet_cfg, len(single_paletler) + 1)
                mix_paletler = mix_palet_data_to_django(mix_palet_data, palet_cfg, optimization)
                optimization.islem_adimi_ekle(f"{len(mix_paletler)} adet mix palet oluşturuldu (Greedy).")

        elif algoritma == 'differential_evolution':
            from src.core.optimizer_de import optimize_with_de, _adaptive_de_params

            optimization.islem_adimi_ekle("Differential Evolution (DE) Motoru ile mix paletler olusturuluyor...")
            optimization.islem_adimi_ekle("İleri seviye optimizasyon teknikleri kullanılıyor...")

            urun_sayisi = len(urun_data_listesi)
            de_base = _adaptive_de_params(urun_sayisi)

            optimization.islem_adimi_ekle(
                f"DE Parametreler: Pop={de_base['population_size']}, "
                f"Nesil={de_base['generations']}, "
                f"Ürün={urun_sayisi}, Fitness Önbellek: Aktif"
            )

            best_chromosome, history = optimize_with_de(
                urunler=urun_data_listesi,
                palet_cfg=palet_cfg,
                F=0.8,
                CR_p=0.9,
            )

            if best_chromosome:
                optimization.islem_adimi_ekle(
                    f"DE En iyi cozum: Fitness={best_chromosome.fitness:.2f}, "
                    f"Palet={best_chromosome.palet_sayisi}, "
                    f"Doluluk={best_chromosome.ortalama_doluluk:.2%}"
                )
                mix_paletler = chromosome_to_palets(
                    best_chromosome,
                    palet_cfg,
                    optimization,
                    len(single_paletler) + 1,
                )
                optimization.islem_adimi_ekle(f"{len(mix_paletler)} adet mix palet oluşturuldu (DE).")
            else:
                optimization.islem_adimi_ekle("DE çözüm üretemedi, Greedy yönteme geçiliyor...")
                mix_palet_data = run_greedy_mix(urun_data_listesi, palet_cfg, len(single_paletler) + 1)
                mix_paletler = mix_palet_data_to_django(mix_palet_data, palet_cfg, optimization)
                optimization.islem_adimi_ekle(f"{len(mix_paletler)} adet mix palet oluşturuldu (Greedy).")

        else:
            optimization.islem_adimi_ekle("Mix paletler oluşturuluyor (Greedy - First-Fit + Maximal Rectangles)...")
            mix_palet_data = run_greedy_mix(urun_data_listesi, palet_cfg, len(single_paletler) + 1)
            mix_paletler = mix_palet_data_to_django(mix_palet_data, palet_cfg, optimization)
            optimization.islem_adimi_ekle(f"{len(mix_paletler)} adet mix palet oluşturuldu (Greedy).")

        mix_motor_sec = round(time.time() - mix_motor_start, 2)
        optimization.islem_adimi_ekle(
            f"[TANI] Mix motoru bitti ({algoritma.upper()}): {mix_motor_sec} sn, "
            f"{len(mix_paletler)} mix palet üretildi"
        )

        # Merge & Repack post-optimizasyon
        merge_start = time.time()
        if len(mix_paletler) >= 2:
            check_cancel(optimization_id, group_id)
            optimization.set_phase('merge1', expected_sec=max(4.0, len(mix_paletler) * 0.8))
            optimization.islem_adimi_ekle(
                f"[1/2] Merge & Repack BFD basliyor "
                f"({len(mix_paletler)} mix palet → iteratif birleştirme)..."
            )
            mix_paletler, mix_metrics = merge_repack_mix_service(
                mix_paletler,
                palet_cfg,
                optimization,
                baslangic_id=len(single_paletler) + 1,
                urun_data_listesi=urun_data_listesi,
            )
            optimization.islem_adimi_ekle(mix_metrics.summary())

            if len(mix_paletler) >= 2:
                check_cancel(optimization_id, group_id)
                optimization.set_phase('merge2', expected_sec=max(3.0, len(mix_paletler) * 0.5))
                optimization.islem_adimi_ekle(
                    f"[2/2] Merge & Repack Random Restart basliyor "
                    f"({len(mix_paletler)} palet kaldı)..."
                )
                mix_paletler, mr_metrics = merge_repack_service(
                    mix_paletler,
                    palet_cfg,
                    optimization,
                    baslangic_id=len(single_paletler) + 1,
                    urun_data_listesi=urun_data_listesi,
                )
                optimization.islem_adimi_ekle(mr_metrics.summary())

        merge_sec = round(time.time() - merge_start, 2)
        optimization.islem_adimi_ekle(
            f"[TANI] Merge & Repack bitti ({algoritma.upper()}): {merge_sec} sn, "
            f"son {len(mix_paletler)} mix palet"
        )

        # Adım 4: İstatistikleri güncelle
        optimization.islem_adimi_ekle("İstatistikler hesaplanıyor...")

        tum_paletler = list(single_paletler) + list(mix_paletler)

        paletler = Palet.objects.filter(optimization=optimization)
        single = paletler.filter(palet_turu='single').count()
        mix = paletler.filter(palet_turu='mix').count()
        optimization.single_palet = single
        optimization.mix_palet = mix
        optimization.toplam_palet = single + mix
        optimization.save()

        son_yerlesmeyen_urunler = []
        for urun in urunler:
            yerlestirilmis = False
            for palet in tum_paletler:
                urun_konumlari = palet.json_to_dict(palet.urun_konumlari)
                if str(urun.id) in urun_konumlari:
                    yerlestirilmis = True
                    break

            if not yerlestirilmis:
                son_yerlesmeyen_urunler.append({
                    'id': urun.id,
                    'urun_kodu': urun.urun_kodu,
                    'urun_adi': urun.urun_adi,
                    'boy': urun.boy,
                    'en': urun.en,
                    'yukseklik': urun.yukseklik,
                    'agirlik': urun.agirlik,
                })

        optimization.yerlesmemis_urunler = son_yerlesmeyen_urunler

        # Tüm paletler için görselleri oluştur
        check_cancel(optimization_id, group_id)
        optimization.set_phase('gorsel', expected_sec=max(2.0, len(tum_paletler) * 0.25))
        optimization.islem_adimi_ekle("Palet görselleri oluşturuluyor...")
        for palet in tum_paletler:
            check_cancel(optimization_id, group_id)
            if not palet.gorsel:
                try:
                    urun_konumlari = palet.json_to_dict(palet.urun_konumlari)
                    urun_ids = [int(id) for id in urun_konumlari.keys()]
                    palet_urunleri = list(Urun.objects.filter(id__in=urun_ids))

                    png_content = palet_gorsellestir(palet, palet_urunleri, save_to_file=True)
                    palet.gorsel.save(f'palet_{palet.palet_id}.png', png_content, save=True)
                    print(f"Palet {palet.palet_id} gorseli olusturuldu")
                except Exception as e:
                    print(f"UYARI: Palet {palet.palet_id} gorseli olusturulamadi: {str(e)}")

        optimization.islem_adimi_ekle("Optimizasyon tamamlandı.")
        optimization.tamamla()

        print(f"\n{'='*60}")
        print(f"OPTIMIZASYON TAMAMLANDI")
        print(f"{'='*60}")
        print(f"Optimization ID: {optimization_id}")
        print(f"Toplam Palet: {optimization.toplam_palet}")
        print(f"Single Palet: {optimization.single_palet}")
        print(f"Mix Palet: {optimization.mix_palet}")
        print(f"Yerleşemeyen: {len(son_yerlesmeyen_urunler)}")
        print(f"{'='*60}\n")

    except OptimizationCancelled:
        print(f"[IPTAL] Optimization {optimization_id} ({algoritma}) kullanıcı tarafından iptal edildi.")
        try:
            optimization = Optimization.objects.get(id=optimization_id)
            optimization.islem_adimi_ekle("İşlem kullanıcı tarafından iptal edildi.")
            durum = optimization.get_islem_durumu()
            durum['current_step'] = -1
            durum['cancelled'] = True
            optimization.islem_durumu = json.dumps(durum)
            optimization.save()
        except Exception as inner_e:
            print(f"Cancel temizlik hatasi: {inner_e}")
        return

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"\n{'='*60}")
        print(f"HATA: OPTIMIZASYON HATASI")
        print(f"{'='*60}")
        print(f"Optimization ID: {optimization_id}")
        print(f"HATA: {str(e)}")
        print(f"DETAY:\n{error_detail}")
        print(f"{'='*60}\n")

        try:
            optimization = Optimization.objects.get(id=optimization_id)
            optimization.islem_adimi_ekle(f"Hata: {str(e)}")
            durum = optimization.get_islem_durumu()
            durum['current_step'] = -1
            optimization.islem_durumu = json.dumps(durum)
            optimization.save()
        except Exception as inner_e:
            print(f"Inner exception: {str(inner_e)}")
