"""DRF serializer'lar — input doğrulama + palet/job çıktı şeması."""

from rest_framework import serializers

from palet_app.models import Palet, Urun


ALGORITHM_CHOICES = ["greedy", "genetic", "differential_evolution"]
GA_MODE_CHOICES = ["fast", "balanced", "quality"]


class ProductSerializer(serializers.Serializer):
    id = serializers.IntegerField(required=False)
    code = serializers.CharField(required=False, allow_blank=True)
    package_length = serializers.FloatField(required=False, allow_null=True, min_value=0)
    package_width = serializers.FloatField(required=False, allow_null=True, min_value=0)
    package_height = serializers.FloatField(required=False, allow_null=True, min_value=0)
    package_weight = serializers.FloatField(required=False, allow_null=True, min_value=0)
    package_in_unit = serializers.FloatField(required=False, allow_null=True)
    package_max_stack_weight = serializers.FloatField(required=False, allow_null=True, min_value=0)
    packages_per_layer = serializers.IntegerField(required=False, allow_null=True)
    unit_length = serializers.FloatField(required=False, allow_null=True, min_value=0)
    unit_width = serializers.FloatField(required=False, allow_null=True, min_value=0)
    unit_height = serializers.FloatField(required=False, allow_null=True, min_value=0)
    unit_weight = serializers.FloatField(required=False, allow_null=True, min_value=0)
    units_per_layer = serializers.IntegerField(required=False, allow_null=True)


class DetailSerializer(serializers.Serializer):
    product = ProductSerializer()
    package_quantity = serializers.IntegerField(required=False, allow_null=True, min_value=0, max_value=100_000)
    quantity = serializers.FloatField(required=False, default=0, min_value=0, max_value=10_000_000)
    unit_id = serializers.CharField(required=False, default="ADET", allow_blank=True)
    package_id = serializers.CharField(required=False, allow_blank=True)

    def validate(self, data):
        pq = data.get("package_quantity")
        qty = data.get("quantity") or 0
        if (pq is None or pq <= 0) and qty <= 0:
            raise serializers.ValidationError(
                "package_quantity veya quantity > 0 olmalı."
            )
        return data


class ContainerSerializer(serializers.Serializer):
    length = serializers.FloatField(min_value=1)
    width = serializers.FloatField(min_value=1)
    height = serializers.FloatField(min_value=1)
    weight = serializers.FloatField(min_value=1)


class OptimizeRequestSerializer(serializers.Serializer):
    """POST /api/v1/optimize/ payload."""

    id = serializers.IntegerField(required=False)
    container = ContainerSerializer()
    details = DetailSerializer(many=True, min_length=1)
    algorithm = serializers.ChoiceField(
        choices=ALGORITHM_CHOICES, required=False, default="greedy"
    )
    ga_mode = serializers.ChoiceField(
        choices=GA_MODE_CHOICES, required=False, default="balanced"
    )
    generate_visuals = serializers.BooleanField(required=False, default=False)

    def validate_details(self, value):
        if len(value) > 5_000:
            raise serializers.ValidationError("details satır sayısı çok fazla (max 5000).")
        return value

    def validate(self, data):
        container = data["container"]
        max_dim = max(container["length"], container["width"], container["height"])
        for idx, detail in enumerate(data["details"]):
            product = detail.get("product", {})
            pq = detail.get("package_quantity")
            if pq and pq > 0:
                dims = [
                    product.get("package_length") or 0,
                    product.get("package_width") or 0,
                    product.get("package_height") or 0,
                ]
            else:
                dims = [
                    product.get("unit_length") or 0,
                    product.get("unit_width") or 0,
                    product.get("unit_height") or 0,
                ]
            largest = max(dims) if dims else 0
            if largest > max_dim:
                raise serializers.ValidationError(
                    f"details[{idx}]: ürün boyutu ({largest}) container'a sığmıyor (max {max_dim})."
                )
        return data


# ---------------- Output ----------------


def _palet_to_dict(palet: Palet, urun_index: dict) -> dict:
    konumlar = palet.json_to_dict(palet.urun_konumlari) or {}
    boyutlar = palet.json_to_dict(palet.urun_boyutlari) or {}

    placements = []
    for uid_str, pos in konumlar.items():
        try:
            uid = int(uid_str)
        except (TypeError, ValueError):
            continue
        u = urun_index.get(uid)
        if not u:
            continue
        b = boyutlar.get(uid_str) or boyutlar.get(str(uid)) or [0, 0, 0]
        try:
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        except (TypeError, ValueError, IndexError):
            x = y = z = 0.0
        try:
            bb, bn, by = float(b[0]), float(b[1]), float(b[2])
        except (TypeError, ValueError, IndexError):
            bb = bn = by = 0.0
        placements.append({
            "urun_id": uid,
            "urun_kodu": u.urun_kodu,
            "urun_adi": u.urun_adi,
            "position": {"x": x, "y": y, "z": z},
            "dimensions": {"boy": bb, "en": bn, "yukseklik": by},
            "agirlik_kg": float(u.agirlik or 0.0),
        })

    try:
        doluluk = round(palet.doluluk_orani(), 2) if palet.hacim() else 0.0
    except (ZeroDivisionError, TypeError, AttributeError):
        doluluk = 0.0

    return {
        "palet_id": palet.palet_id,
        "palet_turu": palet.palet_turu,
        "dimensions": {
            "boy_cm": float(palet.boy),
            "en_cm": float(palet.en),
            "max_yukseklik_cm": float(palet.max_yukseklik),
        },
        "max_agirlik_kg": float(palet.max_agirlik),
        "toplam_agirlik_kg": float(palet.toplam_agirlik or 0.0),
        "kullanilan_hacim_cm3": float(palet.kullanilan_hacim or 0.0),
        "toplam_hacim_cm3": float(palet.hacim() or 0.0),
        "doluluk_orani": doluluk,
        "urun_sayisi": len(placements),
        "urunler": placements,
    }


def serialize_paletler(paletler) -> list:
    """Palet listesini bir kerede serialize eder; tek Urun query'siyle.

    `paletler` bir QuerySet veya iterable olabilir.
    """
    paletler_list = list(paletler)
    if not paletler_list:
        return []

    all_uids = set()
    for p in paletler_list:
        konumlar = p.json_to_dict(p.urun_konumlari) or {}
        for k in konumlar.keys():
            try:
                all_uids.add(int(k))
            except (TypeError, ValueError):
                continue

    urun_index = {}
    if all_uids:
        for u in Urun.objects.filter(id__in=list(all_uids)):
            urun_index[u.id] = u

    return [_palet_to_dict(p, urun_index) for p in paletler_list]
