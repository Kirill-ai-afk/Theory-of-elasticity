from typing import Iterable  # типы анотаций
import numpy as np
import matplotlib.pyplot as plt

from models.material_point import MaterialPoint
from models.body import SquareBody
from models.velocity_field import VelocityField


def plot_trajectories(
        trajectories: Iterable[tuple[MaterialPoint, np.ndarray, np.ndarray]],
        body: SquareBody,
        t0: float,
        t1: float):
    plt.figure(figsize=(6, 6))

    # Начальная форма тела:
    verts = body.vertices()  # вершины
    plt.plot(verts[:, 0], verts[:, 1], "k--", label="Начальная форма тела")

    # Траектории:
    for p0, t, xs in trajectories:
        plt.plot(xs[:, 0], xs[:, 1], "-", linewidth=0.8)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"Траектории материальных точек\n(t от {t0} до {t1})")
    plt.axis("equal")
    plt.grid(True)
    plt.legend(loc="upper right")


def plot_body_deformation(trajectories: Iterable[tuple[MaterialPoint, np.ndarray, np.ndarray]], body: SquareBody, ):
    plt.figure(figsize=(6, 6))

    verts = body.vertices()  # вершины
    plt.plot(verts[:, 0], verts[:, 1], "k--", label="Начальная форма тела")

    # Конечное положение точек:
    end_points = np.array([xs[-1] for _, _, xs in trajectories])  # массив конечных положений всех точек
    plt.scatter(end_points[:, 0], end_points[:, 1], s=10, c="r", label="Точки продеформированного тела")

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Начальное и продеформированное тело")
    plt.axis("equal")
    plt.grid(True)
    plt.legend(loc="upper right")


def plot_velocity_field_and_streamlines(
        field: VelocityField,
        t_star: float,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        grid_n: int = 15):
    # Сетка:
    x = np.linspace(x_min, x_max, grid_n)
    y = np.linspace(y_min, y_max, grid_n)

    """
    X - двумерный массив, в каждой строке повторяется весь массив x;
    Y - двумерный массив, в каждом столбце повторяется весь массив y.
    """
    X, Y = np.meshgrid(x, y)

    U = np.zeros_like(X)  # массив хранения компонент V1
    V = np.zeros_like(Y)  # массив хранения компонент V2

    for i in range(grid_n):
        for j in range(grid_n):
            v = field.v(t_star, np.array([X[i, j], Y[i, j]]))
            U[i, j] = v[0]
            V[i, j] = v[1]

    plt.figure(figsize=(6, 6))

    # 1) Векторное поле:
    plt.quiver(X, Y, U, V, color="gray", label="Поле скоростей")

    # 2) Линии тока:
    plt.streamplot(X, Y, U, V, color="b", linewidth=1.0, arrowsize=1)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"Поле скоростей и линии тока\n(t = {t_star})")
    plt.axis("equal")
    plt.grid(True)
    plt.legend(loc="upper right")


def plot_principal_strain_field(X: np.ndarray, lambda_max: np.ndarray):
    """
    X: сетка начальных координат формы (n, n, 2),
    lambda_max: поле максимальных главных деформаций формы (n, n).
    """

    X1 = X[:, :, 0]
    X2 = X[:, :, 1]

    # Карта главных деформаций:
    plt.figure(figsize=(6, 5))
    levels = 20  # число уровней для контурной заливки
    map_of_strain = plt.contourf(X1, X2, lambda_max, levels=levels, cmap="viridis")
    plt.colorbar(map_of_strain, label="Максимальные главные деформации")
    plt.xlabel("Начальная координата X1")
    plt.ylabel("Начальная координата X2")
    plt.title("Максимальные главные деформации тензора Альманси")
    plt.axis("equal")
    plt.grid(True)
