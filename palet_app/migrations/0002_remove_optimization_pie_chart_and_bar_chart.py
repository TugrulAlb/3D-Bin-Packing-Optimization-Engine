# Grafikler artik Plotly HTML olarak uretiliyor; ImageField'lar kullanilmiyor.

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('palet_app', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(model_name='optimization', name='pie_chart'),
        migrations.RemoveField(model_name='optimization', name='bar_chart'),
    ]
