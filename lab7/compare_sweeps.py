import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lab7 import EPS, LX, LY, errors, sequential_sweep


ORDERS = {
    "x →, y ↑": (False, False),
    "x ←, y ↑": (True, False),
    "x →, y ↓": (False, True),
    "x ←, y ↓": (True, True),
}


def solve_with_order(nx, ny, reverse_x, reverse_y, eps=EPS,
                     max_iterations=1000000):
    if nx < 2 or ny < 2 or eps <= 0 or max_iterations < 1:
        raise ValueError("Нужны nx, ny >= 2, eps > 0, max_iterations >= 1")
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
    history = []
    for _ in range(max_iterations):
        old = u.copy()
        sequential_sweep(u, ax, ay, diagonal, rhs, 1, reverse_x=reverse_x, reverse_y=reverse_y)
        change = np.max(np.abs(u - old))
        history.append(change)
        if not np.isfinite(change):
            raise RuntimeError("Итерации расходятся: изменение не конечно")
        if change <= eps:
            return x, y, u, np.array(history)
    raise RuntimeError(f"Обход не сошёлся за {max_iterations} итераций; "
                       f"изменение {change:.6e}")


def compare(sizes):
    records = []
    for n in sizes:
        print(f"Расчёт четырёх обходов на сетке {n}×{n}…", flush=True)
        for name, (reverse_x, reverse_y) in ORDERS.items():
            x, y, u, history = solve_with_order(n, n, reverse_x, reverse_y)
            record = dict(n=n, order=name, iterations=len(history), change=float(history[-1]),
                          error=float(np.max(errors(x, y, u))))
            records.append(record)

    totals = {name: sum(r["iterations"] for r in records if r["order"] == name) for name in ORDERS}
    minimum = min(totals.values())
    winners = [name for name, total in totals.items() if total == minimum]
    print(f"\nЛучший обход по суммарному числу итераций: {', '.join(winners)}")
    for n in sizes:
        subset = [r for r in records if r["n"] == n]
        best = min(subset, key=lambda r: r["iterations"])
        worst = max(subset, key=lambda r: r["iterations"])
        gain = 100 * (worst["iterations"] - best["iterations"]) / worst["iterations"]
        print(f"N={n}: лучший {best['order']}, выигрыш относительно самого медленного обхода по итерациям {gain:.2f}%")
    return records


def plot_comparison(records, sizes):
    fig, ax = plt.subplots(figsize=(9, 5))
    best = {n: min(r["iterations"] for r in records if r["n"] == n) for n in sizes}
    for name in ORDERS:
        subset = sorted((r for r in records if r["order"] == name), key=lambda r: r["n"])
        relative = [100 * r["iterations"] / best[r["n"]] for r in subset]
        ax.plot([r["n"] for r in subset], relative, 'o-', label=name)
    ax.set(xlabel="Число интервалов N по каждой оси", ylabel="Число итераций, % от лучшего обхода",
           title=f"Эффективность обходов Зейделя при max |u^(k+1) − u^k| <= {EPS:g}\n")
    ax.set_xticks(sizes)
    ax.grid(True, alpha=0.4)
    ax.legend(title="Направления по x и y")
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[10, 20, 40])
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    sizes = sorted(set(args.sizes))
    records = compare(sizes)
    folder = Path(__file__).resolve().parent / "plots"
    folder.mkdir(exist_ok=True)
    fig = plot_comparison(records, sizes)
    path = folder / "sweep_comparison.png"
    fig.savefig(path, dpi=150)
    print(f"График сохранён: {path}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
