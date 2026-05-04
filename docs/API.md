# 3D Bin-Packing REST API (v1)

Harici sistemlerin optimizasyon motoruna iş yollayıp sonuç almasını sağlar.
Tek başına çalışır; mevcut web UI'dan bağımsızdır, aynı algoritma motorunu paylaşır.

## Çalıştırma

```powershell
# 1) Bağımlılıklar (DRF zaten kurulu)
pip install djangorestframework

# 2) API key tanımla (label:key,label2:key2)
$env:BINPACK_API_KEYS = "musteri1:cok-gizli-anahtar-32byte"

# 3) (Opsiyonel) eşzamanlı iş limiti
$env:API_MAX_CONCURRENT_JOBS = "4"

# 4) Sunucu
python manage.py runserver 0.0.0.0:8000
```

İstemci her istekte `X-API-Key: cok-gizli-anahtar-32byte` header'ı yollar. `health` hariç hepsi auth ister.

---

## Endpoint'ler

### `GET /api/v1/health/`
Liveness. Auth gerektirmez. `{"status":"ok", ...}` döner.

### `POST /api/v1/optimize/`
Yeni optimizasyon başlatır. `data/samples/0114.json` ile aynı şema + algoritma seçimi.

**İstek**:
```json
{
  "container": {"length": 120, "width": 100, "height": 180, "weight": 12500},
  "details": [
    {
      "product": {
        "code": "15013726",
        "package_length": 40, "package_width": 30,
        "package_height": 25, "package_weight": 10,
        "package_max_stack_weight": 200,
        "unit_length": 11, "unit_width": 10,
        "unit_height": 12, "unit_weight": 1.003
      },
      "package_quantity": 25,
      "quantity": 339,
      "unit_id": "KG"
    }
  ],
  "algorithm": "genetic",
  "ga_mode": "balanced"
}
```

`algorithm` seçenekleri: `greedy` (default), `genetic`, `differential_evolution`.
`ga_mode` (sadece genetic için): `fast | balanced | quality`.

**Yanıt** `202 Accepted`:
```json
{
  "job_id": 173,
  "status": "queued",
  "algorithm": "genetic",
  "ga_mode": "balanced",
  "product_count": 95,
  "container": {...},
  "links": {
    "status": "/api/v1/optimize/173/status/",
    "result": "/api/v1/optimize/173/result/",
    "cancel": "/api/v1/optimize/173/cancel/"
  },
  "submitted_at": "2026-05-04T10:23:11Z"
}
```

Hatalar: `400 VALIDATION_ERROR`, `401` (key yok), `403` (geçersiz key), `429 CAPACITY_EXCEEDED`.

### `GET /api/v1/optimize/{job_id}/status/`
İş durumu. Tipik polling: 1-2 sn aralıkla.

```json
{
  "job_id": 173,
  "status": "running",         // queued | running | completed | failed | cancelled
  "completed": false,
  "phase": "mix",
  "phase_label": "Mix palet optimizasyonu",
  "percent": 47,
  "messages": ["...", "..."],
  "elapsed_sec": 12.4,
  "started_at": "2026-05-04T10:23:11Z",
  "finished_at": null
}
```

`completed` olduğunda yanıt ek alanlar içerir:
```json
{
  "status": "completed",
  "summary": {"toplam_palet": 7, "single_palet": 2, "mix_palet": 5,
              "yerlesmemis_urun_sayisi": 0},
  "result_url": "/api/v1/optimize/173/result/"
}
```

### `GET /api/v1/optimize/{job_id}/result/`
Tam sonuç. Sadece `completed` ise döner; aksi halde `409 NOT_READY`.

```json
{
  "job_id": 173,
  "status": "completed",
  "algorithm": "genetic",
  "container": {...},
  "summary": {
    "toplam_palet": 7, "single_palet": 2, "mix_palet": 5,
    "single_palet_oran": 28.57, "mix_palet_oran": 71.43,
    "yerlesmemis_urun_sayisi": 0,
    "elapsed_sec": 31.7,
    "started_at": "...", "finished_at": "..."
  },
  "paletler": [
    {
      "palet_id": 1,
      "palet_turu": "single",
      "dimensions": {"boy_cm": 120, "en_cm": 100, "max_yukseklik_cm": 180},
      "max_agirlik_kg": 1250, "toplam_agirlik_kg": 980.5,
      "kullanilan_hacim_cm3": 1620000, "toplam_hacim_cm3": 2160000,
      "doluluk_orani": 75.0,
      "urun_sayisi": 72,
      "urunler": [
        {"urun_id": 142, "urun_kodu": "15013726", "urun_adi": "15013726",
         "position": {"x": 0.0, "y": 0.0, "z": 0.0},
         "dimensions": {"boy": 40.0, "en": 30.0, "yukseklik": 25.0},
         "agirlik_kg": 10.0}
      ]
    }
  ],
  "yerlesmemis_urunler": []
}
```

`position` orijini palet köşesi; tüm uzunluklar cm, ağırlıklar kg.

### `POST /api/v1/optimize/{job_id}/cancel/`
İptal sinyali. Worker bir sonraki faz sınırında durur. `GET` de kabul edilir.
İş zaten bittiyse `409 ALREADY_TERMINAL` döner.

---

## Hata zarfı

Tüm hatalar:
```json
{"error": {"code": "VALIDATION_ERROR", "message": "...", "details": {...}}}
```

Kodlar: `VALIDATION_ERROR`, `NOT_AUTHENTICATED`, `AUTHENTICATION_FAILED`, `NOT_READY`, `ALREADY_TERMINAL`, `CAPACITY_EXCEEDED`, `THROTTLED`, `INTERNAL_ERROR`.

---

## Hız limitleri (varsayılan)

Per API-key:
- submit: 60/dk
- status: 600/dk
- result: 120/dk
- cancel: 60/dk

`API_THROTTLE_SUBMIT` vb. env var'larla override edilebilir. Aşımda `429`.

---

## Örnek curl akışı

```bash
KEY="cok-gizli-anahtar-32byte"
BASE="http://localhost:8000/api/v1"

# 1) İş başlat
JOB=$(curl -s -X POST "$BASE/optimize/" \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  --data @data/samples/0114.json | jq -r .job_id)
echo "Job: $JOB"

# 2) Bitene kadar polla
while true; do
  R=$(curl -s "$BASE/optimize/$JOB/status/" -H "X-API-Key: $KEY")
  STATUS=$(echo "$R" | jq -r .status)
  echo "$(echo $R | jq -r .percent)%  $STATUS"
  [ "$STATUS" = "completed" ] && break
  [ "$STATUS" = "failed" ] && exit 1
  sleep 2
done

# 3) Sonuç al
curl -s "$BASE/optimize/$JOB/result/" -H "X-API-Key: $KEY" > result.json
```

## Python istemci

```python
import time, requests

BASE = "http://localhost:8000/api/v1"
H = {"X-API-Key": "cok-gizli-anahtar-32byte"}

with open("data/samples/0114.json", encoding="utf-8") as f:
    payload = f.read()

resp = requests.post(f"{BASE}/optimize/",
                     headers={**H, "Content-Type": "application/json"},
                     data=payload).json()
job_id = resp["job_id"]

while True:
    s = requests.get(f"{BASE}/optimize/{job_id}/status/", headers=H).json()
    print(f"{s['percent']}% {s['status']}")
    if s["status"] in ("completed", "failed", "cancelled"):
        break
    time.sleep(2)

result = requests.get(f"{BASE}/optimize/{job_id}/result/", headers=H).json()
for palet in result["paletler"]:
    print(f"Palet {palet['palet_id']} ({palet['palet_turu']}): "
          f"{palet['urun_sayisi']} ürün, doluluk %{palet['doluluk_orani']}")
```

---

## Notlar / Sınırlar

- Worker thread, Django process'i ile aynı süreçte çalışır. Üretimde `gunicorn --workers 1 --threads 4` ile başlat (cancel registry tek-process'e bağlı; multi-worker tasarımı v2 işi).
- Maksimum `details` satırı: 5000. Tek SKU için açılan adet: 500.000. Üstü `400` döner.
- `package_quantity` null ise `quantity` (ve `unit_id == "KG"` ise `unit_weight`) okunur.
- 3D render API tarafında üretilmez; istemci kendi görselleştirmesini `position`/`dimensions`'tan yapar.
