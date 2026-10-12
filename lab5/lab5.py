from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


L = np.pi / 2
T = 1.0
N = 40
METHODS = {"Явная": 0.0, "Неявная": 1.0, "Кранка–Николсона": 0.5}
BOUNDARIES = {1: "2 точки, O(h)", 2: "3 точки, O(h^2)",
              3: "2 точки, O(h^2), с уравнением"}


def analytical_solution(x, t):
    return np.sin(t) * np.cos(x)


def source(x, t):
    return np.cos(x) * (np.cos(t) + np.sin(t))


def phi(t):
    return -np.sin(t)


def sweep(a, b, c, d):
    """Метод прогонки (5.31)"""
    n = len(d)
    p = np.zeros(n)
    q = np.zeros(n)
    p[0] = -c[0] / b[0]
    q[0] = d[0] / b[0]
    for i in range(1, n):
        denominator = b[i] + a[i] * p[i - 1]
        p[i] = -c[i] / denominator
        q[i] = (d[i] - a[i] * q[i - 1]) / denominator
    u = np.zeros(n)
    u[-1] = q[-1]
    for i in range(n - 2, -1, -1):
        u[i] = p[i] * u[i + 1] + q[i]
    return u


def solve(n, steps, theta, boundary, end_time=T):
    if n < 3 or steps < 1 or end_time <= 0:
        raise ValueError("Нужно n >= 3, steps >= 1, end_time > 0")
    if theta not in (0, 0.5, 1) or boundary not in BOUNDARIES:
        raise ValueError("Неизвестная схема или аппроксимация границы")
    h = L / n
    tau = end_time / steps
    r = tau / h**2
    if theta == 0 and r > 0.5 + 1e-12:
        raise ValueError("Явная схема требует tau/h^2 <= 1/2")
    x = np.linspace(0, L, n + 1)
    t = np.linspace(0, end_time, steps + 1)
    u = np.zeros((steps + 1, n + 1))

    for k in range(steps):
        old = u[k]
        f = (1 - theta) * source(x, t[k]) + theta * source(x, t[k + 1])
        d = np.zeros(n + 1)
        d[1:n] = (old[1:n] + (1 - theta) * r *
                  (old[:n-1] - 2 * old[1:n] + old[2:]) + tau * f[1:n])
        a = np.full(n + 1, -theta * r)
        b = np.full(n + 1, 1 + 2 * theta * r)
        c = np.full(n + 1, -theta * r)
        a[0], b[0], c[0], d[0] = 0, 1, 0, np.sin(t[k + 1])
        c[n] = 0

        if boundary == 1:
            a[n], b[n], d[n] = -1, 1, h * phi(t[k + 1])
        elif boundary == 2:
            if theta == 0:
                u[k + 1, 0] = d[0]
                u[k + 1, 1:n] = d[1:n]
                u[k + 1, n] = (2*h*phi(t[k + 1]) + 4*d[n-1] - d[n-2]) / 3
                continue
            factor = 1 / a[n - 1]
            a[n] = -4 - factor * b[n - 1]
            b[n] = 3 - factor * c[n - 1]
            d[n] = 2*h*phi(t[k + 1]) - factor * d[n - 1]
        else:
            a[n], b[n] = -2 * theta * r, 1 + 2 * theta * r
            g = (1 - theta) * phi(t[k]) + theta * phi(t[k + 1])
            d[n] = (old[n] + 2*(1-theta)*r*(old[n-1]-old[n])
                    + 2*tau*g/h + tau*f[n])

        if theta == 0:
            u[k + 1, :n] = d[:n]
            u[k + 1, n] = (d[n] - a[n]*u[k + 1, n-1]) / b[n]
        else:
            u[k + 1] = sweep(a, b, c, d)
    return x, t, u


def errors(x, t, u):
    return np.max(np.abs(u - analytical_solution(x[None, :], t[:, None])), axis=1)


def grid_study():
    ns = [10, 20, 40, 80]
    fixed_steps = int(np.ceil(T / (0.4 * (L / max(ns))**2)))
    time_steps = [fixed_steps // 8, fixed_steps // 4,
                  fixed_steps // 2, fixed_steps]
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    for row, (name, theta) in enumerate(METHODS.items()):
        for boundary, label in BOUNDARIES.items():
            eh, et = [], []
            for n in ns:
                x, t, u = solve(n, fixed_steps, theta, boundary)
                eh.append(errors(x, t, u)[-1])
            for steps in time_steps:
                if theta == 0 and T / steps / (L / N)**2 > 0.5:
                    et.append(np.nan)
                    continue
                x, t, u = solve(N, steps, theta, boundary)
                et.append(errors(x, t, u)[-1])
            axes[row, 0].loglog(L / np.array(ns), eh, 'o-', label=label)
            axes[row, 1].loglog(T / np.array(time_steps), et, 'o-', label=label)
        for col, xlabel in enumerate(["h (при постоянном τ)", "τ (при постоянном h)"]):
            ax = axes[row, col]
            ax.set(title=name, xlabel=xlabel, ylabel="max |u(x,T) − U(x,T)|")
            ax.grid(True, which="both")
            ax.legend(fontsize=8)
    fig.suptitle("Зависимость погрешности от шагов сетки, T = 1")
    fig.tight_layout()
    return fig


def main():
    steps = int(np.ceil(T / (0.4 * (L / N)**2)))
    print(f"Вариант 6: N={N}, K={steps}, h={L/N:.6f}, τ={T/steps:.6f}")
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    fig_err, err_axes = plt.subplots(1, 3, figsize=(15, 4))
    for row, (name, theta) in enumerate(METHODS.items()):
        for col, (boundary, label) in enumerate(BOUNDARIES.items()):
            x, t, u = solve(N, steps, theta, boundary)
            error = errors(x, t, u)
            ax = axes[row, col]
            print(f"\n{name}, {label}")
            for moment in [0.25*T, 0.5*T, T]:
                k = int(round(moment / T * steps))
                line, = ax.plot(x, u[k], label=f"Численно, t={t[k]:.3f}")
                ax.plot(x, analytical_solution(x, t[k]), '--', color=line.get_color())
                print(f"t={t[k]:.6f}: max |u−U| = {error[k]:.6e}")
            ax.set(title=f"{name}\n{label}", xlabel="x", ylabel="u(x,t)")
            ax.grid()
            ax.legend(fontsize=7)
            err_axes[row].plot(t, error, label=label)
        err_axes[row].set(title=name, xlabel="t", ylabel="max |u−U|")
        err_axes[row].grid()
        err_axes[row].legend(fontsize=8)
    fig.suptitle("Вариант 6: сплошные линии — численно, пунктир — точное решение")
    fig.tight_layout()
    fig_err.tight_layout()
    fig_grid = grid_study()
    folder = Path(__file__).resolve().parent / "plots"
    folder.mkdir(exist_ok=True)
    for figure, filename in [(fig, "solutions.png"), (fig_err, "errors.png"),
                             (fig_grid, "grid_study.png")]:
        figure.savefig(folder / filename, dpi=150)
    print(f"\nГрафики сохранены в {folder}")
    print("При уменьшении τ погрешность может выйти на плато из-за ошибки по x.")
    plt.show()


if __name__ == "__main__":
    main()
