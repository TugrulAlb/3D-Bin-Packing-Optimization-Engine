"""
Mix Palet Yerleştirme Algoritması
====================================

Farklı boyuttaki ürünleri Genetik Algoritma ile optimum şekilde
paletlere yerleştirir.

Bu modül Django'dan bağımsızdır - saf veri yapıları döndürür.
"""

from .genetic_algorithm import run_ga
from .packing_first_fit import pack_maximal_rectangles_first_fit
from ..models.container import PaletConfig


def mix_palet_yerlestirme_main(mix_pool, palet_cfg: PaletConfig, start_id=1):
    """
    Mix havuzundaki ürünleri Genetik Algoritma ile yerleştirir.
    
    Args:
        mix_pool: UrunData listesi (karışık boyutlar)
        palet_cfg: PaletConfig nesnesi
        start_id: Başlangıç palet ID'si
        
    Returns:
        list[dict]: Her biri 'id', 'type', 'quantity', 'items', 'fill_ratio', 'weight' 
                    içeren palet dict listesi
    """
    if not mix_pool:
        print("Mix havuzu boş, işlem yapılmayacak.")
        return []

    print(f"\n--- Mix Palet (GA) Başlıyor. Ürün Sayısı: {len(mix_pool)} ---")
    
    # 1. Genetik Algoritmayı Çalıştır
    best_solution, history = run_ga(
        urunler=mix_pool,
        palet_cfg=palet_cfg,
        population_size=40,
        generations=50,
        elitism=4,
        mutation_rate=0.2
    )
    
    # 2. En İyi Çözümü Decode Et
    final_pallets_data = pack_maximal_rectangles_first_fit(best_solution.urunler, palet_cfg)
    
    # 3. Sonuçları Formatla
    mix_pallets = []
    current_id = start_id
    
    for p_data in final_pallets_data:
        mix_pallets.append({
            "id": current_id,
            "type": "MIX",
            "quantity": len(p_data["items"]),
            "items": p_data["items"],
            "fill_ratio": p_data.get("fill_ratio", 0),
            "weight": p_data.get("weight", 0)
        })
        current_id += 1
        
    print(f"--- Mix Bitti. GA Tarafından {len(mix_pallets)} adet palet oluşturuldu. ---")
    print(f"En İyi Fitness: {best_solution.fitness:.2f}, "
          f"Ort. Doluluk: %{best_solution.ortalama_doluluk*100:.1f}")
    
    return mix_pallets
