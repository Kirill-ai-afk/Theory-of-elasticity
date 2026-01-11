from analysis.strain_almansi import StrainAlmansiCalculator
from vizualization.plots import plot_principal_strain_field
from models.body import SquareBody
from models.velocity_field import VelocityField
from simulation.trajectory import TrajectorySimulator
from vizualization.plots import (plot_trajectories, plot_body_deformation, plot_velocity_field_and_streamlines)
import matplotlib.pyplot as plt


def main():
    # 1. Тело: квадрат со стороной 2 в, например, 3‑й четверти
    body = SquareBody(side=2.0, center=(-2.0, -2.0))
    points = body.initial_points(n_per_side=6)  # 36 точек

    # 2. Поле скоростей
    field = VelocityField()

    # 3. Траектории точек (t > 0, т.к. ln(t))
    t0 = 1.0
    t1 = 2.0
    dt = 0.01

    traj_sim = TrajectorySimulator(field)
    trajectories = traj_sim.integrate_body(points, t0, t1, dt)

    # 4. Графики
    plot_trajectories(trajectories, body, t0, t1)
    plot_body_deformation(trajectories, body)
    plot_velocity_field_and_streamlines(
        field,
        t_star=1.5,  # фиксированный момент времени для линий тока
        x_min=-4.0,
        x_max=0.0,
        y_min=-4.0,
        y_max=0.0,
        grid_n=15)

    # 5. Тензор деформаций Альманси и максимальные главные деформации
    n_per_side = 11
    almansi_calc = StrainAlmansiCalculator(body, field)

    X_grid, x_grid = almansi_calc.compute_displacement_grid(
        n_per_side=n_per_side,
        t0=t0,
        t_target=2.0,  # t = 2
        dt=dt)

    F_grid = almansi_calc.compute_F(X_grid, x_grid)
    A_grid, lambda_max = almansi_calc.compute_almansi_and_principal(F_grid)

    plot_principal_strain_field(X_grid,lambda_max)

    plt.show()


if __name__ == "__main__":
    main()
