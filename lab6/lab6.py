from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


L = np.pi / 2
T = 1.0
N = 40
METHODS = {"Явная (крест)": 0, "Неявная": 1}
INITIAL_ORDERS = {1: "Начальное условие O(τ)",
                  2: "Начальное условие O(τ^2)"}


def analytical_solution(x, t):
    return np.exp(-x) * np.cos(x) * np.cos(2 * t)


def initial_displacement(x):
    return np.exp(-x) * np.cos(x)


def initial_velocity(x):
    return np.zeros_like(x)


def initial_acceleration(x):
    return -4 * initial_displacement(x)


def sweep(a, b, c, d):
    """Метод прогонки (5.31)"""
    n = len(d)
    p, q = np.zeros(n), np.zeros(n)
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


def solve(n, steps, theta, initial_order, end_time=T):
    """Схемы (5.38), (5.39); theta=0 - крест, theta=1 - неявная"""
    if (not isinstance(n, (int, np.integer)) or
            not isinstance(steps, (int, np.integer)) or
            n < 2 or steps < 1 or not np.isfinite(end_time) or end_time <= 0):
        raise ValueError("int n >= 2, int steps >= 1, int end_time > 0")
    if theta not in METHODS.values() or initial_order not in INITIAL_ORDERS:
        raise ValueError("Неизвестная схема или порядок начального условия")
    h = L / n
    tau = end_time / steps
    if theta == 0 and tau**2 * (1 / h**2 + 0.5) > 1:
        raise ValueError("Не выполняется условие τ^2(1/h^2 + 1/2) <= 1")

    x = np.linspace(0, L, n + 1)
    t = np.linspace(0, end_time, steps + 1)
    u = np.zeros((steps + 1, n + 1))
    u[0] = initial_displacement(x)
    u[1] = u[0] + tau * initial_velocity(x)
    if initial_order == 2:
        u[1] += tau**2 / 2 * initial_acceleration(x)
    u[:, 0] = np.cos(2 * t)
    u[:, -1] = 0

    left = tau**2 * (1 / h**2 - 1 / h)
    center = tau**2 * (-2 / h**2 - 2)
    right = tau**2 * (1 / h**2 + 1 / h)
    a = np.full(n + 1, -theta * left)
    b = np.full(n + 1, 1 - theta * center)
    c = np.full(n + 1, -theta * right)
    a[0], b[0], c[0] = 0, 1, 0
    a[n], b[n], c[n] = 0, 1, 0

    for k in range(1, steps):
        d = 2 * u[k] - u[k - 1]
        if theta == 0:
            u[k + 1, 1:n] = (d[1:n] + left * u[k, :n-1]
                              + center * u[k, 1:n] + right * u[k, 2:])
        else:
            d[0], d[n] = u[k + 1, 0], u[k + 1, n]
            u[k + 1] = sweep(a, b, c, d)
    return x, t, u


def errors(x, t, u):
    """Максимальная по пространству абсолютная погрешность на каждом слое"""
    return np.max(np.abs(u - analytical_solution(x[None, :], t[:, None])), axis=1)


def grid_study():
    ns = np.array([10, 20, 40, 80])
    fixed_steps = 2048
    time_steps = np.array([64, 128, 256, 512])
    fixed_n = 80
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for row, (name, theta) in enumerate(METHODS.items()):
        for order, label in INITIAL_ORDERS.items():
            eh, et = [], []
            for n in ns:
                x, t, u = solve(n, fixed_steps, theta, order)
                eh.append(errors(x, t, u)[-1])
            for steps in time_steps:
                x, t, u = solve(fixed_n, steps, theta, order)
                et.append(errors(x, t, u)[-1])
            axes[row, 0].loglog(L / ns, eh, 'o-', label=label)
            axes[row, 1].loglog(T / time_steps, et, 'o-', label=label)
            print(f"\n{name}, {label}: исследование сетки")
            for h, error in zip(L / ns, eh):
                print(f"h={h:.6f}, τ={T/fixed_steps:.6f}: ошибка(T)={error:.6e}")
            for tau, error in zip(T / time_steps, et):
                print(f"h={L/fixed_n:.6f}, τ={tau:.6f}: ошибка(T)={error:.6e}")
        for col, xlabel in enumerate(["h (при постоянном τ)", "τ (при постоянном h)"]):
            axes[row, col].set(title=name, xlabel=xlabel,
                               ylabel="max |u(x,T) − U(x,T)|")
            axes[row, col].grid(True, which="both")
            axes[row, col].legend(fontsize=8)
    fig.suptitle(f"Зависимость погрешности от шагов сетки, T = {T:g}")
    fig.tight_layout()
    return fig


def main():
    steps = int(np.ceil(T / (0.8 * (L / N))))
    print(f"Вариант 6: N={N}, K={steps}, h={L/N:.6f}, τ={T/steps:.6f}")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig_err, err_axes = plt.subplots(1, 2, figsize=(12, 4))
    for row, (name, theta) in enumerate(METHODS.items()):
        for col, (order, label) in enumerate(INITIAL_ORDERS.items()):
            x, t, u = solve(N, steps, theta, order)
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
            ax.legend(fontsize=8)
            err_axes[row].plot(t, error, label=label)
        err_axes[row].set(title=name, xlabel="t", ylabel="max |u−U|")
        err_axes[row].grid()
        err_axes[row].legend(fontsize=8)
    fig.suptitle("Сплошные линии — численно, пунктир — точное решение")
    fig.tight_layout()
    fig_err.tight_layout()
    fig_grid = grid_study()
    folder = Path(__file__).resolve().parent / "plots"
    folder.mkdir(exist_ok=True)
    for figure, filename in [(fig, "solutions.png"), (fig_err, "errors.png"),
                             (fig_grid, "grid_study.png")]:
        figure.savefig(folder / filename, dpi=150)
    print(f"\nГрафики сохранены в {folder}")
    print("При уменьшении одного шага ошибка может выйти на плато из-за другого.")
    plt.show()


if __name__ == "__main__":
    main()
