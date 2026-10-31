#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализ измерений скорости генерации LLM.

Читает data/measurements.csv, строит линейную регрессию
    dt = a + b * chars
по всем точкам с нагрузкой (kind = trial), печатает наклон, свободный член,
ошибки, R^2 и перевод в токены/с при разных допущениях о символах на токен.

Запуск:  python analysis/analyze.py
"""
import csv
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "data", "measurements.csv")

# Эмпирический диапазон для связного русского текста: 1 токен ~= 3..4 символа
CHARS_PER_TOKEN = (3.0, 3.5, 4.0)


def load_trials(path):
    chars, dts, labels = [], [], []
    baseline_dt = None
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["point"] == "baseline":
                baseline_dt = float(row["dt_seconds"])
                continue
            chars.append(float(row["chars"]))
            dts.append(float(row["dt_seconds"]))
            labels.append(f'опыт {row["experiment"]}, {row["point"]}')
    return np.array(chars), np.array(dts), labels, baseline_dt


def fit(x, y):
    n = len(x)
    b, a = np.polyfit(x, y, 1)
    res = y - (a + b * x)
    ss_res = float((res ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    s_err = np.sqrt(ss_res / (n - 2)) / np.sqrt(((x - x.mean()) ** 2).sum())
    return a, b, s_err, r2, res


def main():
    x, y, labels, baseline_dt = load_trials(CSV_PATH)

    print("Точки измерений:")
    for lab, c, d in zip(labels, x, y):
        print(f"  {lab:16s}  {int(c):5d} симв  {d:8.3f} с")
    if baseline_dt:
        print(f"\nХолостой заезд опыта 1 (оценка накладных расходов): {baseline_dt:.2f} с")

    a, b, s_err, r2, res = fit(x, y)
    cps = 1.0 / b
    cps_err = s_err / b ** 2

    print("\n== Линейная регрессия dt = a + b * chars ==")
    print(f"наклон b              = {b * 1000:.2f} ± {s_err * 1000:.2f} мс/символ")
    print(f"свободный член a      = {a:.1f} с (накладные расходы на заезд)")
    print(f"R^2                   = {r2:.4f}")
    print(f"остатки (с)           = {[round(float(r), 1) for r in res]}")
    print(f"\nскорость потока       = {cps:.1f} ± {cps_err:.1f} символов/с")

    print("\nПеревод в токены/с (допущение о символах на токен — эвристика!):")
    for cpt in CHARS_PER_TOKEN:
        print(f"  при {cpt} симв/токен: {cps / cpt:.1f} ± {cps_err / cpt:.1f} ток/с")

    # Перекрёстная проверка: метод вычитания холостого заезда (только опыт 1)
    if baseline_dt:
        mask = np.array(["опыт 1" in l for l in labels])
        net = y[mask][0] - baseline_dt
        print(f"\nПерекрёстная проверка (опыт 1, вычитание холостого заезда):")
        print(f"  чистое время {net:.1f} с на {int(x[mask][0])} симв -> "
              f"{x[mask][0] / net:.1f} символов/с (в пределах ошибки регрессии: "
              f"{abs(x[mask][0] / net - cps) <= cps_err})")


if __name__ == "__main__":
    main()
