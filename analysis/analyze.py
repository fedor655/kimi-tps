#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализ измерений скорости генерации LLM.

Читает data/measurements.csv и строит линейные регрессии
    dt = a + b * chars
ОТДЕЛЬНО по сессиям измерений (смешивать разные заходы в одну прямую
некорректно: у каждой сессии свои накладные расходы и своя загрузка кластера):

  сессия A (2026-10-31, вечер)   — эксперименты 1 и 2, 5 точек
  сессия B (2026-11-01, ночь, сообщено: Kimi K2.8) — эксперимент 3, 3 точки

Запуск:  python analysis/analyze.py
"""
import csv
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "data", "measurements.csv")

# Эксперимент -> сессия
SESSIONS = {
    "A (2026-10-31, вечер, десктопный клиент)": {"1", "2"},
    "B (2026-11-01, ночь, сообщено: Kimi K2.8)": {"3"},
}

# Эмпирический диапазон для связного русского текста: 1 токен ~= 3..4 символа
CHARS_PER_TOKEN = (3.0, 3.5, 4.0)


def load(path):
    rows = []
    baseline_dt = None
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["point"] == "baseline":
                baseline_dt = float(row["dt_seconds"])
                continue
            rows.append({
                "experiment": row["experiment"],
                "point": row["point"],
                "chars": float(row["chars"]),
                "dt": float(row["dt_seconds"]),
            })
    return rows, baseline_dt


def fit(x, y):
    n = len(x)
    b, a = np.polyfit(x, y, 1)
    res = y - (a + b * x)
    ss_res = float((res ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    if n > 2:
        s_err = np.sqrt(ss_res / (n - 2)) / np.sqrt(((x - x.mean()) ** 2).sum())
    else:
        s_err = float("nan")
    return a, b, s_err, r2, res


def report(name, rows):
    x = np.array([r["chars"] for r in rows])
    y = np.array([r["dt"] for r in rows])
    a, b, s_err, r2, res = fit(x, y)
    cps, cps_err = 1.0 / b, s_err / b ** 2

    print(f"\n{'=' * 66}\nСессия {name} — {len(rows)} точек")
    for r, e in zip(rows, res):
        print(f"  опыт {r['experiment']}, {r['point']:8s} {int(r['chars']):5d} симв  "
              f"{r['dt']:8.2f} с   остаток {e:+.2f} с")
    print(f"  наклон b        = {b * 1000:.2f} ± {s_err * 1000:.2f} мс/символ")
    print(f"  свободный член  = {a:.2f} с (накладные расходы на заезд)")
    print(f"  R^2             = {r2:.4f}")
    print(f"  скорость потока = {cps:.1f} ± {cps_err:.1f} символов/с")
    for cpt in CHARS_PER_TOKEN:
        print(f"    при {cpt} симв/токен: {cps / cpt:.1f} ± {cps_err / cpt:.1f} ток/с")
    return cps, cps_err, a


def main():
    rows, baseline_dt = load(CSV_PATH)
    if baseline_dt:
        print(f"Холостой заезд сессии A (оценка накладных расходов): {baseline_dt:.2f} с")

    results = {}
    for name, exps in SESSIONS.items():
        sel = [r for r in rows if r["experiment"] in exps]
        results[name] = report(name, sel)

    # Сравнение сессий
    (cps_a, err_a, a_a), (cps_b, err_b, a_b) = results.values()
    print(f"\n{'=' * 66}\nСравнение сессий")
    print(f"  скорость:  {cps_a:.1f} ± {err_a:.1f}  vs  {cps_b:.1f} ± {err_b:.1f} симв/с"
          f"  ({(cps_b / cps_a - 1) * 100:+.0f} %)")
    print(f"  накладные расходы: {a_a:.1f} с  vs  {a_b:.1f} с")
    # Перекрёстная проверка сессии A методом вычитания холостого заезда
    if baseline_dt:
        t1 = next(r for r in rows if r["experiment"] == "1")
        net = t1["dt"] - baseline_dt
        cps_net = t1["chars"] / net
        print(f"\nПерекрёстная проверка сессии A (опыт 1, вычитание холостого заезда):")
        print(f"  чистое время {net:.1f} с на {int(t1['chars'])} симв -> {cps_net:.1f} симв/с "
              f"(в пределах ошибки регрессии: {abs(cps_net - cps_a) <= err_a})")


if __name__ == "__main__":
    main()
