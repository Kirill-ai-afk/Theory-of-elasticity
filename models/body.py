import numpy as np
from .material_point import MaterialPoint


class Body:

    def __init__(self):
        self.center = None  # длина стороны квадрата/размера
        self.side = None  # центр тела (np.array([x0, y0]))

    def grid_points(self, n_per_side: int) -> np.ndarray:

        """
        Сетка начальных координат X формы (n_per_side, n_per_side, 2) - трехмерный массив,
        где X[i, j, :] = (X1, X2) - точка со своей координатой
        """

        # Половина длины стороны квадрата:
        half = self.side / 2.0

        # Одномерный массив из n_per_side значений по оси X1 от левой границы квадрата до правой:
        xs = np.linspace(self.center[0] - half, self.center[0] + half, n_per_side)

        # Одномерный массив из n_per_side значений по оси X2 от нижней границы квадрата до верхней:
        ys = np.linspace(self.center[1] - half, self.center[1] + half, n_per_side)

        X = np.zeros((n_per_side, n_per_side, 2))
        for i in range(len(xs)):
            for j in range(len(ys)):
                X[i, j, 0] = xs[i]  # X1
                X[i, j, 1] = ys[j]  # X2

        return X


class SquareBody(Body):
    """
    Квадрат со стороной side и центром в точке center.
    Дано: side = 2, центр выбираем в нужной четверти.
    """

    def __init__(self, side: float = 2.0, center: tuple[float, float] = (int, int)):
        super().__init__()
        self.side = side
        self.center = np.array(center)

    def initial_points(self, n_per_side: int):

        """
        Возвращает список объектов типа "MaterialPoint" внутри квадрата:
        равномерная сетка n_per_side × n_per_side.
        """

        # Половина длины стороны квадрата:
        half = self.side / 2.0

        # Одномерный массив из n_per_side значений по оси X1 от левой границы квадрата до правой:
        xs = np.linspace(self.center[0] - half, self.center[0] + half, n_per_side)

        # Одномерный массив из n_per_side значений по оси X2 от нижней границы квадрата до верхней:
        ys = np.linspace(self.center[1] - half, self.center[1] + half, n_per_side)

        points = []
        for x in xs:
            for y in ys:
                points.append(MaterialPoint(np.array([x, y])))
        return points

    def vertices(self) -> np.ndarray:
        """
        Координаты вершин квадрата (для рисования контура).
        Пять точек: первая = последняя, чтобы получить замкнутую линию.
        """

        # Половина длины стороны квадрата:
        half = self.side / 2.0

        x0, y0 = self.center
        verts = np.array([[x0 - half, y0 - half],
                          [x0 + half, y0 - half],
                          [x0 + half, y0 + half],
                          [x0 - half, y0 + half],
                          [x0 - half, y0 - half],])
        return verts
