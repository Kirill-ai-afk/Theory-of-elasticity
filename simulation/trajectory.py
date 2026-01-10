from typing import Iterable, List, Tuple  # типы анотаций
import numpy as np

from models.material_point import MaterialPoint
from models.velocity_field import VelocityField
from numerics.runge_kutta import RungeKuttaButcher


class TrajectorySimulator:
    """
    Строит траектории материальных точек в заданном поле скоростей
    методом Рунге–Кутты.
    """

    def __init__(self, field: VelocityField):
        self.field = field

    def _rhs(self, t: float, x: np.ndarray) -> np.ndarray:
        """Правая часть ОДУ: dx/dt = v(t, x)."""
        return self.field.v(t, x)

    def integrate_point(self, point: MaterialPoint, t0: float, t1: float, dt: float, ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Траектория одной точки.
        Возвращает массив времени t и массив координат x(t) формы (N, 2).
        """
        # Объект класса Рунге-Кутты:
        rk = RungeKuttaButcher(self._rhs)
        steps = int((t1 - t0) / dt)  # количество шагов

        t = np.linspace(t0, t1, steps + 1)  # распределенный массив времени
        xs = np.zeros((steps + 1, 2))  # массив точек (+1 - для начального момента времени)
        xs[0] = point.x  # обращаемся к атрибуту класса MaterialPoint

        cur_t = t0
        cur_x = point.x.copy()  # копирование атрибута

        for i in range(steps):
            cur_x = rk.step(cur_t, cur_x, dt)
            cur_t += dt
            xs[i + 1] = cur_x

        return t, xs

    def integrate_body(self, points: Iterable[MaterialPoint], t0: float, t1: float, dt: float, ) \
            -> List[tuple[MaterialPoint, np.ndarray, np.ndarray]]:
        """
        Траектории для набора точек тела.
        Возвращает список (исходная_точка, t, x(t)).
        """
        result: list[tuple[MaterialPoint, np.ndarray, np.ndarray]] = []
        for p in points:
            t, xs = self.integrate_point(p, t0, t1, dt)
            result.append((p, t, xs))
        return result

