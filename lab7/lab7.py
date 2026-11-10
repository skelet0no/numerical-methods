from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


LX = np.pi / 2
LY = 1
NX = NY = 40
EPS = 1e-8
OMEGA = 1.7
METHODS = {"Либмана": "jacobi", "Зейделя": "seidel", "Верхняя релаксация": "sor"}


def analytical_solution(x, y):
    return y * np.sin(x)


def sequential_sweep(u, ax, ay, diagonal, rhs, relaxation,
                     reverse_x=True, reverse_y=True):
    ny, nx = u.shape[0] - 1, u.shape[1] - 1
    xs = range(nx - 1, 0, -1) if reverse_x else range(1, nx)
    ys = range(ny, -1, -1) if reverse_y else range(ny + 1)
    values = u.tolist()
    forcing = rhs.tolist()
    denominators = diagonal[:, 0].tolist()
    for j in ys:
        row = values[j]
        below = values[j - 1] if j > 0 else values[0]
        above = values[j + 1] if j < ny else values[ny]
        x_weight = ax if 0 < j < ny else 0
        below_weight = ay if 0 < j < ny else int(j == ny)
        above_weight = ay if 0 < j < ny else int(j == 0)
        denominator = denominators[j]
        row_rhs = forcing[j]
        for i in xs:
            neighbours = (x_weight * (row[i - 1] + row[i + 1]) + below_weight * below[i] + above_weight * above[i])
            candidate = (neighbours + row_rhs[i - 1]) / denominator
            row[i] += relaxation * (candidate - row[i])
    u[:] = values


def solve(nx, ny, method, eps=EPS, omega=OMEGA,
          max_iterations=1000000):
    x = np.linspace(0, LX, nx + 1)
    y = np.linspace(0, LY, ny + 1)
    hx, hy = LX / nx, LY / ny
    ax, ay = 1 / hx**2, 1 / hy**2
    diagonal = np.full((ny + 1, 1), 2 * ax + 2 * ay - 1)
    diagonal[0] = 1
    diagonal[-1] = 1 - hy
    rhs = np.zeros((ny + 1, nx - 1))
    rhs[0] = -hy * np.sin(x[1:-1])

    u = y[:, None] * x[None, :] / LX
    u[:, 0] = 0
    u[:, -1] = y
    relaxation = omega if method == "sor" else 1
    history = []

    def neighbours(v):
        result = ax * (v[:, :-2] + v[:, 2:])
        result[1:-1] += ay * (v[:-2, 1:-1] + v[2:, 1:-1])
        result[0] = v[1, 1:-1]
        result[-1] = v[-2, 1:-1]
        return result

    for _ in range(max_iterations):
        old = u.copy()
        if method == "jacobi":
            u[:, 1:-1] = (neighbours(u) + rhs) / diagonal
        else:
            sequential_sweep(u, ax, ay, diagonal, rhs, relaxation)
        change = np.max(np.abs(u - old))
        history.append(change)
        if not np.isfinite(change):
            raise RuntimeError("Итерации расходятся: изменение не конечно")
        if change <= eps:
            return x, y, u, np.array(history)
    raise RuntimeError(f"Метод {method} не сошёлся за {max_iterations} итераций; "
                       f"изменение {change:.6e}")


def errors(x, y, u):
    return np.abs(u - analytical_solution(x[None, :], y[:, None]))


def grid_study():
    ns = np.array([5, 10, 20, 40])
    fixed_n = 40
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, method in METHODS.items():
        for col in range(2):
            values = []
            for n in ns:
                nx, ny = (n, fixed_n) if col == 0 else (fixed_n, n)
                x, y, u, history = solve(nx, ny, method)
                error = np.max(errors(x, y, u))
                values.append(error)
                print(f"{name}: hx={LX/nx:.6f}, hy={LY/ny:.6f}, ошибка={error:.6e}, итераций={len(history)}")
            steps = (LX if col == 0 else LY) / ns
            axes[col].loglog(steps, values, 'o-', label=name)
        print("")
    for ax, xlabel in zip(axes, ["hx (при постоянном hy)", "hy (при постоянном hx)"]):
        ax.set(xlabel=xlabel, ylabel="max |u − U|")
        ax.grid(True, which="both")
        ax.legend(fontsize=8)
    fig.suptitle("Зависимость погрешности от шагов сетки")
    fig.tight_layout()
    return fig


def main():
    print(f"ЛР 7: Nx={NX}, Ny={NY}, hx={LX/NX:.6f}, hy={LY/NY:.6f}")
    print(f"Точность по изменению приближений: {EPS:g}; omega={OMEGA:g}")
    print("Обход Зейделя и SOR: справа налево, сверху вниз")
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig_err, err_axes = plt.subplots(1, 3, figsize=(15, 4))
    fig_conv, conv_ax = plt.subplots(figsize=(8, 5))
    for row, (name, method) in enumerate(METHODS.items()):
        x, y, u, history = solve(NX, NY, method)
        error = errors(x, y, u)
        print(f"\n{name}: итераций={len(history)}, изменение={history[-1]:.6e}, max |u−U|={np.max(error):.6e}")
        for fraction in [0.25, 0.5, 1]:
            j, i = round(fraction * NY), round(fraction * NX)
            line, = axes[row, 0].plot(x, u[j], label=f"y={y[j]:.3f}")
            axes[row, 0].plot(x, analytical_solution(x, y[j]), '--', color=line.get_color())
            line, = axes[row, 1].plot(y, u[:, i], label=f"x={x[i]:.3f}")
            axes[row, 1].plot(y, analytical_solution(x[i], y), '--', color=line.get_color())
        for col, xlabel in enumerate(["x", "y"]):
            axes[row, col].set(title=name, xlabel=xlabel, ylabel="u(x,y)")
            axes[row, col].grid()
            axes[row, col].legend(fontsize=8)
        mesh = err_axes[row].pcolormesh(x, y, error, shading="auto")
        err_axes[row].set(title=name, xlabel="x", ylabel="y")
        fig_err.colorbar(mesh, ax=err_axes[row], label="|u − U|")
        conv_ax.semilogy(np.arange(1, len(history) + 1), history, label=name)
    fig.suptitle("Сплошные линии — численно, пунктир — точное решение")
    fig.tight_layout()
    fig_err.tight_layout()
    conv_ax.set(xlabel="Номер итерации", ylabel="max |u^(k+1) − u^k|", title="Сходимость итерационных методов")
    conv_ax.grid(True, which="both")
    conv_ax.legend()
    fig_conv.tight_layout()
    print("\nИсследование сетки:\n")
    fig_grid = grid_study()
    folder = Path(__file__).resolve().parent / "plots"
    folder.mkdir(exist_ok=True)
    for figure, filename in [(fig, "solutions.png"), (fig_err, "errors.png"), (fig_conv, "convergence.png"),
                             (fig_grid, "grid_study.png")]:
        figure.savefig(folder / filename, dpi=150)
    print(f"Графики сохранены в {folder}")
    plt.show()


if __name__ == "__main__":
    main()
