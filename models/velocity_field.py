import numpy as np


class VelocityField:
    """
    Инициализация поля скоростей:
    v1 = -e^t * x1
    v2 = -ln(t) * x2,  t > 0
    """

    def __init__(self):
        self.v1 = None
        self.v2 = None

    def v(self, t: float, x: np.ndarray) -> np.ndarray:
        self.v1 = -np.exp(t) * x[0]
        self.v2 = -np.log(t) * x[1]
        return np.array([self.v1, self.v2])
