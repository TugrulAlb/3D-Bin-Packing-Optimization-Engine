# 3D Bin Packing Optimization Engine

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![Django 4.2.7](https://img.shields.io/badge/Django-4.2.7-darkgreen)](https://www.djangoproject.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 About the Project

**3D Bin Packing Optimization Engine** is a professional optimization solution developed to solve the NP-Hard 3D bin packing problem using **Genetic Algorithm (GA)** approach.

Core objectives of the project:
- ✅ Place products in containers optimally
- ✅ Minimize empty space (maximize container utilization)
- ✅ Ensure weight balance
- ✅ Consider physical constraints and rotation rules
- ✅ Provide web-based interactive interface

## 🎯 Key Features

### Algorithm Engine
- **Genetic Algorithm (GA)**: Population-based evolutionary optimization
- **Fitness Function**: Container utilization rate + weight distribution
- **Maximal Rectangles Packing**: Guillotine-based 2D/3D placement
- **Single Pallet Algorithm**: Grid-based placement validation
- **Mixed Product Optimization**: Intelligent product grouping

### Web Interface (Django)
- 📁 Product and container management
- 🎨 3D visualization (Matplotlib + Plotly)
- 📊 Optimization results and analysis reports
- 💾 Historical tracking of optimized operations
- 🔄 JSON data import/export

## 📦 Realistic Packing (Amazon-like)

Gerçek depo istifine benzer kompakt, katmanlı ve stabil yerleşimler için
dört ek mekanizma etkinleştirilmiştir:

| Mekanizma | Dosya | Ne yapar? |
|---|---|---|
| **Void Penalty** | `src/core/fitness.py` | Bounding-box hacmi ile gerçek kutu hacmi farkını ölçer; büyük iç boşluklar (U şekli, oyuklar) ceza alır. |
| **Layer Snapping** | `src/core/packing.py` | Kutu z koordinatı, mevcut katman yüzeylerine (layer_map) veya Z_GRID=5 cm ızgarasına yuvarlanır. Raf gibi temiz katman görünümü sağlar. |
| **Edge Bias** | `src/core/fitness.py` | Ürünler duvarlara ne kadar yakınsa o kadar ödüllendirilir; kenar boşlukları azalır. |
| **Cavity Penalty** | `src/core/fitness.py` | XY ayak izindeki kapalı iç boşluklar (baca kolonları) flood-fill ile tespit edilir ve cezalandırılır. N=4 throttle ile performans korunur. |

### Parametreler (`src/core/fitness.py` başı)

```python
W_VOID        = 0.8    # Void ceza ağırlığı        [0.6 – 1.2]
W_EDGE        = 0.15   # Kenar ödül ağırlığı       [0.1 – 0.3]
W_CAVITY      = 0.35   # Cavity ceza ağırlığı      [0.2 – 0.6]
CAVITY_GRID   = 5.0    # Cavity grid adımı (cm)
CAVITY_THROTTLE = 4    # Her N bireyde cavity hesapla
```

`Z_GRID` (katman snap adımı cm) için `src/core/packing.py` dosyasının başına bakın.

## 🏗️ Project Structure

```
3D-Bin-Packing-Optimization-Engine/
├── src/                           # Main algorithm library
│   ├── core/
│   │   ├── genetic_algorithm.py   # GA implementation
│   │   ├── fitness.py             # Fitness calculation
│   │   ├── chromosome.py          # Chromosome representation
│   │   ├── single_pallet.py       # Single pallet algorithm
│   │   ├── packing.py             # Rectangle packing
│   │   └── mix_pallet.py          # Mixed product optimization
│   ├── models/
│   │   ├── product.py             # Product data model
│   │   └── container.py           # Container data model
│   └── utils/
│       ├── parser.py              # JSON input parser
│       ├── helpers.py             # Helper functions
│       └── visualization.py       # 3D visualization
│
├── palet_app/                     # Django application
│   ├── models/
│   │   ├── palet.py               # Pallet ORM model
│   │   ├── urun.py                # Product ORM model
│   │   └── optimization.py        # Optimization results
│   ├── views.py                   # Django views
│   ├── urls.py                    # URL routing
│   ├── services.py                # Business logic layer
│   └── templates/                 # HTML templates
│
├── core/                          # Django project config
│   ├── settings.py                # Django settings
│   ├── urls.py                    # URL configuration
│   └── wsgi.py                    # WSGI entry point
│
├── data/samples/                  # Test JSON files
├── output/                        # Output directory (images, reports)
├── templates/                     # Base HTML templates
├── main.py                        # Standalone CLI entry point
├── manage.py                      # Django management
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Installation & Running

### Requirements
- Python 3.11+ (tested on Python 3.12.2)
- pip (Python package manager)
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/TugrulAlb/3D-Bin-Packing-Optimization-Engine.git
cd 3D-Bin-Packing-Optimization-Engine
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file (copy from `.env.example`):

```bash
cp .env.example .env
```

**Generate a secure SECRET_KEY:**

```bash
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

Edit `.env` and set:
- `SECRET_KEY`: Your generated secret key
- `DEBUG`: `True` for development, `False` for production
- `ALLOWED_HOSTS`: Comma-separated list (e.g., `localhost,127.0.0.1`)
- `DEBUG_SUPPORT`: Set to `1` to enable detailed optimization logging

### 5. Database Setup

```bash
python manage.py migrate
```

### 6. Run Development Server

```bash
python manage.py runserver
```

Visit: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

---

## 🛠️ Development Setup

### Python Version
- **Required**: Python 3.11+
- **Tested**: Python 3.12.2

### Environment Variables

The application supports the following environment variables (see `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `SECRET_KEY` | Django secret key (required for production) | Development key |
| `DEBUG` | Enable debug mode | `True` |
| `ALLOWED_HOSTS` | Comma-separated allowed hosts | `localhost,127.0.0.1` |
| `DEBUG_SUPPORT` | Enable detailed support constraint logging | `0` |
| `MEDIA_ROOT` | Media files directory | `media` |
| `STATIC_ROOT` | Static files directory | `staticfiles` |

**Security Note**: Never commit `.env` with real secrets to version control!

### Running Optimization Algorithms

**Genetic Algorithm (GA)**:
```bash
python main.py data/samples/0110.json
```

**Differential Evolution (DE)**:
```python
from src.core.optimizer_de import run_de
from src.models.container import PaletConfig

# Configure and run
palet_cfg = PaletConfig(length=120, width=100, height=150, max_weight=1000)
best_solution, history = run_de(
    urunler=products,
    palet_cfg=palet_cfg,
    population_size=80,  # Auto: max(60, 0.8*N)
    generations=50,
    use_rotations=False
)
```

### Testing

Run optimization tests:
```bash
# Test gravity constraint
DEBUG_SUPPORT=1 python test_gravity_constraint.py

# Test productionization features
DEBUG_SUPPORT=1 python test_productionization.py
```

### Algorithm Selection

The web interface supports both optimization algorithms:
- **Genetic Algorithm (GA)**: Traditional evolutionary approach
- **Differential Evolution (DE)**: Advanced hybrid mutation strategy with Amazon-style stability constraints

**Key Parameters**:
- **GA**: Population size, generations, mutation rate, crossover rate
- **DE**: NP (min: max(60, 0.8×N)), generations, F (adaptive 0.4-0.9), CR (0.9)
- **Gravity Constraint**: min_support_ratio = 0.40 (40% support required above ground)

---

##  Input Data Format (JSON)

```json
{
  "containers": [
    {
      "id": "container_1",
      "length": 1200,
      "width": 800,
      "height": 1000,
      "max_weight": 5000
    }
  ],
  "products": [
    {
      "id": "prod_001",
      "code": "SKU-12345",
      "length": 100,
      "width": 80,
      "height": 50,
      "weight": 20,
      "quantity": 5,
      "rotatable": true
    }
  ]
}
```

## 📈 Output Data Format

Optimization results are stored in `output/reports/` in JSON format:

```json
{
  "containers": [
    {
      "container_id": "container_1",
      "utilization": 0.78,
      "weight_balance": 0.92,
      "placements": [
        {
          "product_id": "prod_001",
          "position": [0, 0, 0],
          "orientation": [100, 80, 50]
        }
      ]
    }
  ],
  "total_utilization": 0.78,
  "total_weight": 4500,
  "execution_time": 2.34
}
```

## 🔧 Basic Usage Examples

### Using Python API

```python
from src.core.genetic_algorithm import run_ga
from src.models import PaletConfig, UrunData

# Define container
container = PaletConfig(length=1200, width=800, height=1000, max_weight=5000)

# Define product
product = UrunData(
    urun_id=1,
    code="SKU-001",
    boy=100, en=80, yukseklik=50,
    agirlik=20,
    quantity=5
)

# Run optimization
result = run_ga(
    containers=[container],
    urunler=[product],
    population_size=50,
    generations=100
)

print(f"Container Utilization: {result['utilization']:.2%}")
```

### Using Django ORM

```python
from palet_app.models import Palet, Urun
from palet_app.services import optimize_pallet

# Get products from database
products = Urun.objects.all()

# Run optimization
result = optimize_pallet(products)

# Save results
palet = Palet.objects.create(
    name=f"Optimized_{result['id']}",
    utilization=result['utilization']
)
```

## 📈 Algorithm Details

### Genetic Algorithm Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Population Size | 50 | Number of individuals in population |
| Generations | 100 | Number of evolution cycles |
| Mutation Rate | 0.1 | Mutation probability (0-1) |
| Crossover Rate | 0.8 | Crossover probability (0-1) |
| Selection Type | Tournament | Tournament selection |

### Fitness Calculation

```
Fitness = (w1 × Utilization) + (w2 × Weight_Balance) - (w3 × Penalty)

where:
- Utilization: Container filling ratio (0-1)
- Weight_Balance: Weight balance index (0-1)  
- Penalty: Constraint violation penalty (0-1)
- w1, w2, w3: Weight coefficients
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test category
python -m pytest tests/algorithms/ -v

# Coverage report
pytest --cov=src tests/
```

## 📦 Dependencies

| Package | Version | Usage |
|---------|---------|-------|
| Django | 4.2.7 | Web framework |
| NumPy | 1.24.3 | Numerical computation |
| Matplotlib | 3.7.1 | 2D visualization |
| Plotly | 5.18.0+ | Interactive 3D charts |
| Pillow | 10.0.0 | Image processing |
| gunicorn | 21.2.0 | Production WSGI server |

All dependencies are listed in `requirements.txt`.

## 🚢 Production Deployment

### Running with Gunicorn

```bash
gunicorn core.wsgi:application --bind 0.0.0.0:8000
```

### Docker (Optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "core.wsgi:application", "--bind", "0.0.0.0:8000"]
```

## 🔗 Related Resources

- [NP-Hard Problem - Wikipedia](https://en.wikipedia.org/wiki/NP-hardness)
- [Bin Packing Problem](https://en.wikipedia.org/wiki/Bin_packing_problem)
- [Genetic Algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm)
- [Django Official Documentation](https://docs.djangoproject.com/)

---

**Happy Optimizing! 🚀**
