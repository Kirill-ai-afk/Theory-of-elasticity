import numpy as np


class RungeKuttaButcher:
    """
    Двухстадийный метод Рунге–Кутты по таблице Бутчера:

        c1 |   0   0
        c2 | a21   0
        --------------
          |  b1    b2

    c1 = 0
    c2 = 1
    a21 = 1
    b1 = 1/2
    b2 = 1/2

    k1 = f(t_n, x_n)
    k2 = f(t_n + с2 * h, x_n + h * (a21 * k1))
    x_{n+1} = x_n + h * (b1 * k1 + b2 * k2)
    """

    c2 = 1
    a21 = 1
    b1 = b2 = 1 / 2

    def __init__(self, f):
        """
        f - функция двух аргументов
        """
        self.f = f

    # Метод Рунге-Кутта второго порядка:
    def step(self, t: float, x: np.ndarray, h: float) -> np.ndarray:  # принимает xi
        k1 = self.f(t, x)
        k2 = self.f(t + RungeKuttaButcher.c2 * h, x + h * (RungeKuttaButcher.a21 * k1))
        return x + h * (RungeKuttaButcher.b1 * k1 + RungeKuttaButcher.b2 * k2)  # возвращает x(i+1)
