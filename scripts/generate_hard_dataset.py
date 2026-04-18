"""GA/DE'yi Greedy'den ayırt etmek için zor sentetik bin-packing dataset'i üretir.

Single pre-pass'i atlatmak için çok sayıda SKU, her SKU'dan az adet üretilir;
boyutlar 3 tier'a bölünür (büyük/orta/küçük) ve palet kapasitesi tam sığacak
şekilde hesaplanır — yani sıralama GA/DE için kritik hale gelir.
"""

import argparse
import json
import math
import os
import random
import sys


SIZE_PROFILES = {
    'small':   {'n_sku': 25,  'min_qty': 2, 'max_qty': 4, 'fill_factor': 0.90},
    'medium':  {'n_sku': 80,  'min_qty': 2, 'max_qty': 4, 'fill_factor': 0.92},
    'large':   {'n_sku': 180, 'min_qty': 2, 'max_qty': 4, 'fill_factor': 0.94},
    'xlarge':  {'n_sku': 400, 'min_qty': 2, 'max_qty': 4, 'fill_factor': 0.95},
}

TIER_SHARES = [
    ('large',  0.30, (40, 60), (35, 55), (30, 55)),
    ('medium', 0.40, (20, 40), (18, 38), (15, 40)),
    ('small',  0.30, (8,  20), (8,  20), (8,  25)),
]

CONTAINER = {
    'length': 120,
    'width':  100,
    'height': 180,
    'weight': 1250,
}


def _tier_dims(tier_ranges, rng):
    _, _, lr, wr, hr = tier_ranges
    length = rng.choice([lr[0], (lr[0] + lr[1]) // 2, lr[1]])
    width = rng.choice([wr[0], (wr[0] + wr[1]) // 2, wr[1]])
    height = rng.choice([hr[0], (hr[0] + hr[1]) // 2, hr[1]])
    return length, width, height


def _tier_weight(tier_name, rng):
    if tier_name == 'large':
        return round(rng.uniform(18.0, 45.0), 2)
    if tier_name == 'medium':
        return round(rng.uniform(4.0, 18.0), 2)
    return round(rng.uniform(0.5, 4.0), 2)


def generate_dataset(size: str, seed: int):
    profile = SIZE_PROFILES[size]
    rng = random.Random(seed)

    container_volume = CONTAINER['length'] * CONTAINER['width'] * CONTAINER['height']

    skus = []
    tier_counts = {}
    for tier in TIER_SHARES:
        tier_name, share, *_ = tier
        n_tier = max(1, int(round(profile['n_sku'] * share)))
        tier_counts[tier_name] = n_tier
        for _ in range(n_tier):
            length, width, height = _tier_dims(tier, rng)
            weight = _tier_weight(tier_name, rng)
            skus.append({
                'length': length,
                'width': width,
                'height': height,
                'weight': weight,
                'tier': tier_name,
            })

    rng.shuffle(skus)

    details = []
    product_id = 30000
    total_volume = 0.0
    total_items = 0

    for sku in skus:
        qty = rng.randint(profile['min_qty'], profile['max_qty'])
        sku_volume = sku['length'] * sku['width'] * sku['height']
        for k in range(qty):
            details.append({
                'product': {
                    'id': product_id,
                    'code': f'SYN{product_id:06d}',
                    'package_length': sku['length'],
                    'package_width':  sku['width'],
                    'package_height': sku['height'],
                    'package_weight': sku['weight'],
                },
                'package_quantity': 1,
                'package_id': f'SYN-{product_id}-{k+1}',
            })
            total_volume += sku_volume
            total_items += 1
        product_id += 1

    target_pallets = max(1, math.ceil(total_volume / (container_volume * profile['fill_factor'])))

    dataset = {
        'id': 90000000 + abs(hash((size, seed))) % 9000000,
        'container': CONTAINER,
        'details': details,
        '_meta': {
            'generator': 'generate_hard_dataset.py',
            'size': size,
            'seed': seed,
            'n_sku': profile['n_sku'],
            'n_items': total_items,
            'tier_counts': tier_counts,
            'container_volume': container_volume,
            'total_item_volume': total_volume,
            'theoretical_min_pallets': target_pallets,
            'expected_fill_factor': profile['fill_factor'],
        },
    }
    return dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', choices=SIZE_PROFILES.keys(), default='medium',
                        help='Dataset boyut tier (default: medium)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Deterministik seed (default: 42)')
    parser.add_argument('--output', default=None,
                        help='Output JSON yolu (default: data/samples/hard_<size>.json)')
    args = parser.parse_args(argv)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output = args.output or os.path.join(
        project_root, 'data', 'samples', f'hard_{args.size}.json'
    )

    dataset = generate_dataset(args.size, args.seed)
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    meta = dataset['_meta']
    print(f"[OK] {output}")
    print(f"  size={meta['size']} seed={meta['seed']}")
    print(f"  n_sku={meta['n_sku']}  n_items={meta['n_items']}")
    print(f"  tier_counts={meta['tier_counts']}")
    print(f"  theoretical_min_pallets={meta['theoretical_min_pallets']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
