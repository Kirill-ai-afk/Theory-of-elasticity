import numpy as np

from models.velocity_field import VelocityField
from numerics.runge_kutta import RungeKuttaButcher


class StreamlineCalculator:
    """
    Строит линии тока при фиксированном времени t_star.
    Использует метод Рунге–Кутты, параметр интегрирования — s.
    """

    def __init__(self, field: VelocityField):
        # Сохраняем поле скоростей v(t, x)
        self.field = field

    def _rhs_stream(self, t_star: float, x: np.ndarray) -> np.ndarray:
        """
        Правая часть ОДУ для линии тока: dx/ds = v(t*, x).
        t_star фиксировано, поэтому функция зависит только от x (и параметра s формально).
        """
        return self.field.v(t_star, x)

    def integrate_streamline(self, x0: np.ndarray, t_star: float, s_max: float, ds: float) -> np.ndarray:
        """
        Строим одну линию тока, исходящую из точки x0, при фиксированном времени t_star.
        s идёт от 0 до s_max с шагом ds.
        Возвращает массив точек xs формы (steps+1, 2).
        """

        # Рунге–Кутта для системы dx/ds = v(t*, x):
        """
        Аргументом является функция двух переменных f(s, x), 
        где s - это "праметр-затычка" для соблюдения количества аргументов.  
        В данной реализации первый аргумент функции f(s, x) (функции наклона k1 и k2) фиксирован и равен t_star,        
        а x - это второй аргумент f(s, x), который принимает уже различные значения в зависимости от k1 и k2  
        """
        rk = RungeKuttaButcher(lambda s, x: self._rhs_stream(t_star, x))

        steps = int(s_max / ds)  # количество шагов по параметру s
        xs = np.zeros((steps + 1, 2))  # сюда пишем точки линии
        xs[0] = x0  # стартовая точка

        cur_s = 0.0  # текущий параметр s (начинаем с 0)
        cur_x = x0.copy()  # текущие координаты точки на линии тока, сначала это копия начальной точки.

        """  
        Чисто гипотетически параметр cur_s используется в алгоритме метода rk.step для k1 и k2, но он игнорируется        
        так как первый аргумент функции f(s, x) у нас фиксирован и равен t_star - cur_s нужно, чтобы соблюсти        
        количество аргументов метода rk.step.  
        """
        for i in range(steps):
            # Один шаг Рунге–Кутты по параметру s
            cur_x = rk.step(cur_s, cur_x, ds)
            cur_s += ds
            xs[i + 1] = cur_x

        return xs

    def multiple_streamlines(self, seeds: list[np.ndarray], t_star: float, s_max: float, ds: float) -> list[np.ndarray]:
        """
        Строит несколько линий тока из набора стартовых точек seeds.
        Возвращает список массивов xs, по одному массиву на каждую линию.
        """
        lines = []
        for x0 in seeds:
            line = self.integrate_streamline(x0, t_star, s_max, ds)
            lines.append(line)  # заполнение списка
        return lines
