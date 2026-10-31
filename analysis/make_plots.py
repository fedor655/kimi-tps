#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Строит графики по данным измерений:
  plots/regression.png — точки, прямая регрессии, 95% доверительная полоса
  plots/residuals.png  — остатки регрессии

Запуск:  python analysis/make_plots.py
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "data", "measurements.csv")
PLOTS = os.path.join(ROOT, "plots")
os.makedirs(PLOTS, exist_ok=True)


def load(path):
    per_exp = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["point"] == "baseline":
                continue
            per_exp.setdefault(row["experiment"], []).append(
                (float(row["chars"]), float(row["dt_seconds"]), row["point"]))
    return per_exp


def main():
    per_exp = load(CSV_PATH)
    x = np.array([p[0] for pts in per_exp.values() for p in pts])
    y = np.array([p[1] for pts in per_exp.values() for p in pts])
    n = len(x)

    b, a = np.polyfit(x, y, 1)
    res = y - (a + b * x)
    s = np.sqrt((res ** 2).sum() / (n - 2))
    x_line = np.linspace(0, x.max() * 1.08, 200)
    y_line = a + b * x_line
    # 95% доверительная полоса линии регрессии (t-критический для n-2=3 степеней свободы)
    t_crit = 3.182
    se_line = s * np.sqrt(1 / n + (x_line - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum())

    # --- График 1: регрессия ---
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"1": "#d1495b", "2": "#00798c"}
    names = {"1": "Опыт 1: одиночная нагрузка", "2": "Опыт 2: регрессионная серия"}
    for exp, pts in per_exp.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.scatter(xs, ys, s=90, color=colors[exp], label=names[exp], zorder=3)
        for px, py, lab in pts:
            ax.annotate(lab, (px, py), textcoords="offset points",
                        xytext=(8, -14), fontsize=9, color=colors[exp])
    ax.fill_between(x_line, y_line - t_crit * se_line, y_line + t_crit * se_line,
                    color="#2e86ab", alpha=0.15, label="95% доверительная полоса")
    ax.plot(x_line, y_line, color="#2e86ab", lw=2,
            label=f"dt = {a:.1f} + {b * 1000:.2f}·L   (R² = {1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum():.3f})")
    ax.set_xlabel("Длина нагрузки L, символов")
    ax.set_ylabel("Время заезда dt, секунды")
    ax.set_title("Время генерации против длины нагрузки\n"
                 f"наклон = {b * 1000:.2f} мс/символ → {1 / b:.1f} символов/с ≈ 16 ток/с")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "regression.png"), dpi=150)
    plt.close(fig)

    # --- График 2: остатки ---
    fig, ax = plt.subplots(figsize=(9, 4))
    for exp, pts in per_exp.items():
        xs = [p[0] for p in pts]
        rs = [pv[1] - (a + b * pv[0]) for pv in pts]
        ax.scatter(xs, rs, s=90, color=colors[exp], label=names[exp], zorder=3)
    ax.axhline(0, color="#2e86ab", lw=1.5)
    ax.set_xlabel("Длина нагрузки L, символов")
    ax.set_ylabel("Остаток, секунды")
    ax.set_title("Остатки регрессии: излома нет, цена символа постоянна вдоль всего диапазона")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "residuals.png"), dpi=150)
    plt.close(fig)

    print("Сохранено: plots/regression.png, plots/residuals.png")


if __name__ == "__main__":
    main()
