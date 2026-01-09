import numpy as np


class MaterialPoint:
    """
    Материальная точка в плоскости.
    x: np.ndarray с координатами (x1, x2).
    """

    def __init__(self, x: np.ndarray):
        self.x = x

    def copy(self) -> "MaterialPoint":
        """
        self.x.copy() создаёт независимую копию массива (метод возвращает
        новую материальную точку с такими же координатами),
        чтобы изменение координат новой точки не влияло на исходную.
        """
        return MaterialPoint(self.x.copy())
