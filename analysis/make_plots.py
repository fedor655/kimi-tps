#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Строит графики по данным измерений (отдельная регрессия для каждой сессии):
  plots/regression.png — точки обеих сессий, свои прямые, 95% доверительные полосы
  plots/residuals.png  — остатки регрессий

Запуск:  python analysis/make_plots.py
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats  # только для t-критического; если scipy нет — см. fallback ниже

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "data", "measurements.csv")
PLOTS = os.path.join(ROOT, "plots")
os.makedirs(PLOTS, exist_ok=True)

# Сессия -> (эксперименты, цвет, подпись)
SESSIONS = {
    "A": ({"1", "2"}, "#00798c", "Сессия A (2026-10-31, вечер)"),
    "B": ({"3"}, "#d1495b", "Сессия B (2026-11-01, сообщено: Kimi K2.8)"),
    "C": ({"4"}, "#2e8540", "Сессия C (2026-11-05, Kimi по API, ZCode)"),
}


def load(path):
    per_session = {k: [] for k in SESSIONS}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["point"] == "baseline":
                continue
            for name, (exps, _, _) in SESSIONS.items():
                if row["experiment"] in exps:
                    per_session[name].append(
                        (float(row["chars"]), float(row["dt_seconds"]), row["point"]))
    return per_session


def fit_with_band(x, y, x_line):
    """Возвращает (a, b, r2, band) — band это полуширина 95% ДП линии на x_line."""
    n = len(x)
    b, a = np.polyfit(x, y, 1)
    res = y - (a + b * x)
    r2 = 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()
    if n > 2:
        s = np.sqrt((res ** 2).sum() / (n - 2))
        try:
            t_crit = stats.t.ppf(0.975, n - 2)
        except Exception:
            t_crit = 3.182
        band = t_crit * s * np.sqrt(1 / n + (x_line - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum())
    else:
        band = np.zeros_like(x_line)
    return a, b, r2, res, band


def main():
    per_session = load(CSV_PATH)

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    for name, (pts, color, label) in ((k, (v, *SESSIONS[k][1:])) for k, v in per_session.items()):
        x = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        x_line = np.linspace(0, x.max() * 1.15, 200)
        a, b, r2, res, band = fit_with_band(x, y, x_line)
        ax.scatter(x, y, s=95, color=color, zorder=3,
                   label=f"{label}: {1 / b:.1f} симв/с ≈ {1 / b / 3.5:.0f} ток/с")
        ax.plot(x_line, a + b * x_line, color=color, lw=2, alpha=0.85,
                label=f"  dt = {a:.1f} + {b * 1000:.1f}·L   (R² = {r2:.3f})")
        if band.any():
            ax.fill_between(x_line, a + b * x_line - band, a + b * x_line + band,
                            color=color, alpha=0.10)
        for px, py, lab in pts:
            ax.annotate(lab, (px, py), textcoords="offset points",
                        xytext=(8, -14), fontsize=9, color=color)
    ax.set_xlabel("Длина нагрузки L, символов")
    ax.set_ylabel("Время заезда dt, секунды")
    ax.set_title("Время генерации против длины нагрузки — три сессии, три прямые")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "regression.png"), dpi=150)
    plt.close(fig)

    # --- График 2: остатки ---
    fig, ax = plt.subplots(figsize=(9.5, 4))
    for name, (pts, color, label) in ((k, (v, *SESSIONS[k][1:])) for k, v in per_session.items()):
        x = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        a, b, r2, res, _ = fit_with_band(x, y, x)
        ax.scatter(x, res, s=95, color=color, label=label, zorder=3)
    ax.axhline(0, color="#555555", lw=1.2)
    ax.set_xlabel("Длина нагрузки L, символов")
    ax.set_ylabel("Остаток, секунды")
    ax.set_title("Остатки регрессий по сессиям: изломов нет, цена символа постоянна")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "residuals.png"), dpi=150)
    plt.close(fig)

    print("Сохранено: plots/regression.png, plots/residuals.png")


if __name__ == "__main__":
    main()
