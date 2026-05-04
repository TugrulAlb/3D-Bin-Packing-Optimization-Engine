"""Konteyner / palet konfigürasyon modeli."""


class PaletConfig:
    """Palet parametreleri (cm, kg)."""
    
    def __init__(self, length, width, height, max_weight):
        self.length = float(length)
        self.width = float(width)
        self.height = float(height)
        self.max_weight = float(max_weight)

    @property
    def volume(self) -> float:
        """Palet toplam hacmi (cm³)."""
        return self.length * self.width * self.height

    def __repr__(self):
        return (f"PaletConfig({self.length}x{self.width}x{self.height}, "
                f"max_weight={self.max_weight})")
