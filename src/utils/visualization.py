"""
Palet Görselleştirme Modülü (Django-Bağımsız)
===============================================

Matplotlib ile 3D palet görselleştirme.
Bu modül herhangi bir web framework'e bağımlı değildir.
"""

import matplotlib
matplotlib.use('Agg')  # GUI olmadan çalışması için
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import io
import random

# Renk havuzu (tutarlılık için)
random.seed(42)
COLOR_MAP = {}


def renk_uret(code):
    """Her ürün kodu için tutarlı renk üretir."""
    if code not in COLOR_MAP:
        COLOR_MAP[code] = (
            random.random() * 0.6 + 0.2,
            random.random() * 0.6 + 0.2,
            random.random() * 0.6 + 0.2
        )
    return COLOR_MAP[code]


def kutu_ciz(ax, x, y, z, dx, dy, dz, color):
    """3D kutu çizer (solid, iç gözükmez)."""
    xx = [x, x, x+dx, x+dx, x, x, x+dx, x+dx]
    yy = [y, y+dy, y+dy, y, y, y+dy, y+dy, y]
    zz = [z, z, z, z, z+dz, z+dz, z+dz, z+dz]
    
    vertices = [
        [0, 1, 2, 3], [4, 5, 6, 7],
        [0, 1, 5, 4], [2, 3, 7, 6],
        [1, 2, 6, 5], [0, 3, 7, 4]
    ]
    
    faces = []
    for v in vertices:
        faces.append([[xx[v[i]], yy[v[i]], zz[v[i]]] for i in range(4)])
    
    poly = Poly3DCollection(faces, alpha=0.9, facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_collection3d(poly)


def render_pallet_3d(palet_boy, palet_en, palet_yukseklik, items, title="Palet"):
    """
    3D palet görselleştirme - Framework bağımsız.
    
    Args:
        palet_boy (float): Palet uzunluğu (cm)
        palet_en (float): Palet genişliği (cm)
        palet_yukseklik (float): Palet yüksekliği (cm)
        items (list[dict]): Her biri 'urun_kodu', 'x', 'y', 'z', 'L', 'W', 'H' içeren dict
        title (str): Grafik başlığı
        
    Returns:
        io.BytesIO: PNG formatında görsel
    """
    fig = plt.figure(figsize=(12, 9), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    PL, PW, PH = palet_boy, palet_en, palet_yukseklik
    
    # Ürünleri çiz
    for item in items:
        renk = renk_uret(item['urun_kodu'])
        kutu_ciz(ax, item['x'], item['y'], item['z'],
                 item['L'], item['W'], item['H'], renk)
    
    # Palet sınırları (kırmızı çerçeve)
    for z_val in [0, PH]:
        ax.plot([0, PL], [0, 0], [z_val, z_val], 'r-', linewidth=2)
        ax.plot([0, PL], [PW, PW], [z_val, z_val], 'r-', linewidth=2)
        ax.plot([0, 0], [0, PW], [z_val, z_val], 'r-', linewidth=2)
        ax.plot([PL, PL], [0, PW], [z_val, z_val], 'r-', linewidth=2)
    
    for x_val, y_val in [(0, 0), (PL, 0), (0, PW), (PL, PW)]:
        ax.plot([x_val, x_val], [y_val, y_val], [0, PH], 'r-', linewidth=2)
    
    ax.set_xlabel('Boy (cm)', fontsize=10)
    ax.set_ylabel('En (cm)', fontsize=10)
    ax.set_zlabel('Yükseklik (cm)', fontsize=10)
    ax.set_xlim([0, PL])
    ax.set_ylim([0, PW])
    ax.set_zlim([0, PH])
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.view_init(elev=20, azim=45)
    
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf


def render_summary_charts(palet_data_list):
    """
    Özet grafikleri oluşturur (Plotly).
    
    Args:
        palet_data_list: list[dict] - Her biri 'palet_id', 'palet_turu', 'doluluk' içeren dict
        
    Returns:
        tuple: (pie_chart_html, bar_chart_html) veya (None, None)
    """
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError:
        return None, None
    
    single_count = sum(1 for p in palet_data_list if p['palet_turu'] == 'single')
    mix_count = sum(1 for p in palet_data_list if p['palet_turu'] == 'mix')
    
    # 1. Pasta grafik
    fig1 = go.Figure(data=[go.Pie(
        labels=['Single', 'Mix'],
        values=[single_count, mix_count],
        hole=0.3,
        marker=dict(colors=['#3498db', '#e74c3c']),
        textinfo='label+percent',
        textfont_size=14
    )])
    fig1.update_layout(
        title=dict(text='Palet Tipi Dağılımı', x=0.5, xanchor='center'),
        height=350,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    pie_chart_html = pio.to_html(fig1, full_html=False, include_plotlyjs='cdn')
    
    # 2. Bar grafik
    ids = [f"P{p['palet_id']}" for p in palet_data_list]
    doluluklar = [p['doluluk'] for p in palet_data_list]
    colors_bar = ['#3498db' if p['palet_turu'] == 'single' else '#e74c3c' for p in palet_data_list]
    
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=ids, y=doluluklar,
        marker_color=colors_bar,
        text=[f'{d:.1f}%' for d in doluluklar],
        textposition='outside',
        textfont_size=10,
        name='Doluluk'
    ))
    fig2.add_hline(y=80, line_dash="dash", line_color="green",
                   annotation_text="Hedef %80", annotation_position="right")
    fig2.update_layout(
        title=dict(text='Palet Doluluk Oranları', x=0.5, xanchor='center'),
        yaxis_title='Doluluk Oranı (%)',
        yaxis_range=[0, 105],
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False
    )
    bar_chart_html = pio.to_html(fig2, full_html=False, include_plotlyjs='cdn')
    
    return pie_chart_html, bar_chart_html
