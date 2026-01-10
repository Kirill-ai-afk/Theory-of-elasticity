import numpy as np
from typing import Tuple  # типы анотаций

from simulation.trajectory import TrajectorySimulator
from models.body import SquareBody
from models.velocity_field import VelocityField
from models.material_point import MaterialPoint


class StrainAlmansiCalculator:
    """
    Вычисляет тензор деформаций Альманси и максимальные главные деформации
    по численным траекториям R(X, t) на сетке X (11x11) в момент времени t_target.
    """

    def __init__(self, body: SquareBody, field: VelocityField):
        self.body = body
        self.field = field

    def compute_displacement_grid(self, n_per_side: int, t0: float, t_target: float, dt: float, ) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Считает движение R: X -> x(X, t_target) для сетки X (n x n x 2).
        Возвращает (X, x), обе формы (n, n, 2).
        """

        # Двумерная равномерная сетка точек в квадрате:
        X = self.body.grid_points(n_per_side)

        traj_sim = TrajectorySimulator(self.field)

        n = n_per_side

        # Массив хранения конечных положений всех точек сетки:
        x = np.zeros_like(X)

        for i in range(n):
            for j in range(n):
                x0 = X[i, j, :]  # начальное положение
                point = MaterialPoint(x0)
                t, xs = traj_sim.integrate_point(point, t0, t_target, dt)
                x[i, j, :] = xs[-1, :]  # берем положение точки в последний момент времени (-1 - последняя строка)

        return X, x

    def compute_F(self, X: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Вычисляет градиент движения F = (grad r)^t = dx/dX (лагранжевый подход) конечными разностями.
        X, x формы (n, n, 2).
        Возвращает F[i,j,:,:] формы (n, n, 2, 2) – матрица dx/dX в узлах.
        """
        n = X.shape[0]  # находим размер массива и берем число узлов по первой коордтнате

        """
        Первые две компоненты - номера узлов ij, 
        а вторые - матрица 2 на 2 ab, где F_{ab} = dx_a/dX_b
        """
        F = np.zeros((n, n, 2, 2))  #

        # Равномерная сетка по X:
        dX1 = X[1, 0, 0] - X[0, 0, 0]  # шаг сетки по координате X1 между соседними узлами по индексу i
        dX2 = X[0, 1, 1] - X[0, 0, 1]  # шаг сетки по координате X2 между соседними узлами по индексу j

        for i in range(1, n - 1):  # цикл по внутренним узлам для центральных разностей (границы не трогаем)
            for j in range(1, n - 1):
                # Центральные разности по X1 (индекс i):
                dx_dX1 = (x[i + 1, j, :] - x[i - 1, j, :]) / (2.0 * dX1)
                # центральные разности по X2 (индекс j)
                dx_dX2 = (x[i, j + 1, :] - x[i, j - 1, :]) / (2.0 * dX2)

                # формируем матрицу градиента F = dx/dX
                F[i, j, 0, 0] = dx_dX1[0]
                F[i, j, 1, 0] = dx_dX1[1]
                F[i, j, 0, 1] = dx_dX2[0]
                F[i, j, 1, 1] = dx_dX2[1]

        return F

    def compute_almansi_and_principal(self, F: np.ndarray, ) -> Tuple[np.ndarray, np.ndarray]:
        """
        По (grad r)^T = F вычисляем тензор Альманси A и его собственные значения.
        F формы (n, n, 2, 2).
        Возвращает:
          A – (n, n, 2, 2),
          lambda_max – (n, n) – максимальная главная деформация.
        """
        n = F.shape[0]  # число узлов
        A = np.zeros_like(F)  # массви тензорра Альманси
        lambda_max = np.zeros((n, n))  # максимальная главная деформация (наибольшее собственное значение A) в каждом
        # узле.

        E = np.eye(2)  # единичная матрица

        for i in range(1, n - 1):  # цикл по внутренним узлам
            for j in range(1, n - 1):
                F_ij = F[i, j, :, :]

                # Ищем F^{-1} для тензора А:
                F_inv = np.linalg.inv(F_ij)

                # Находим g = F^{-T} * F^{-1} - мера деформации Альманси
                g = F_inv.T @ F_inv  # перемножение для согласованных матриц
                A_ij = 0.5 * (E - g)  # A = 1/2 * (E - g) = 1/2 * (E - F^{-T} * F^{-1})

                A[i, j, :, :] = A_ij

                # Собственные значения (максимальная главная деформация):
                vals = np.linalg.eigvals(A_ij)
                lambda_max[i, j] = np.max(np.real(vals))

        return A, lambda_max
