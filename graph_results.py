"""
Gráficas generadas (en screenshots/):
  1. bar_times_per_operation.png  — tiempo promedio CPU vs GPU por operación
  2. speedup_per_operation.png    — speedup por operación (barras)
  3. transfer_overhead.png        — desglose GPU: operación vs transferencia
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # sin pantalla (útil en servidores)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_DIR    = os.path.join(os.path.dirname(__file__), "results")
SCREENSHOTS_DIR = os.path.join(os.path.dirname(__file__), "screenshots")

# Paleta
C_CPU    = "#E05C5C"   # rojo
C_GPU    = "#4A90D9"   # azul
C_XFER   = "#A8D0F0"   # azul claro (transferencia)
C_SPEED  = "#2ECC71"   # verde (speedup)
FONT     = "DejaVu Sans"

plt.rcParams.update({
    "font.family":       FONT,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
})


def load_data():
    path = os.path.join(RESULTS_DIR, "comparison.csv")
    if not os.path.isfile(path):
        print(f"ERROR: '{path}' no encontrado. Ejecuta comparison.py primero.")
        exit(1)
    return pd.read_csv(path)


def ms(col):
    """Convierte columna de segundos a milisegundos."""
    return col * 1000


def save(fig, name):
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    path = os.path.join(SCREENSHOTS_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardada: {path}")


# ── Gráfica 1: Tiempos promedio por operación ─────────────────────────────────
def plot_bar_times(df):
    ops    = ["Inversión", "Gauss. Blur", "Ecualización", "Total"]
    cpu_ms = [
        ms(df["cpu_invert"]).mean(),
        ms(df["cpu_gaussian"]).mean(),
        ms(df["cpu_equalize"]).mean(),
        ms(df["cpu_total"]).mean(),
    ]
    gpu_ms = [
        ms(df["gpu_invert"]).mean(),
        ms(df["gpu_gaussian"]).mean(),
        ms(df["gpu_equalize"]).mean(),
        ms(df["gpu_total"]).mean(),
    ]

    x  = np.arange(len(ops))
    w  = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))

    bars_cpu = ax.bar(x - w/2, cpu_ms, w, color=C_CPU, label="CPU", zorder=3)
    bars_gpu = ax.bar(x + w/2, gpu_ms, w, color=C_GPU, label="GPU (sin transferencia)", zorder=3)

    for bar in bars_cpu:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8, color=C_CPU)
    for bar in bars_gpu:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8, color=C_GPU)

    ax.set_xticks(x)
    ax.set_xticklabels(ops)
    ax.set_ylabel("Tiempo promedio (ms)")
    ax.set_title("Tiempo de procesamiento promedio por operación\nCPU vs GPU")
    ax.legend()
    save(fig, "bar_times_per_operation.png")


# ── Gráfica 2: Speedup por operación ─────────────────────────────────────────
def plot_speedup_ops(df):
    ops = ["Inversión", "Gauss. Blur", "Ecualización", "Total (GPU pura)", "Total (GPU + xfer)"]
    speedups = [
        df["speedup_invert"].mean(),
        df["speedup_gaussian"].mean(),
        df["speedup_equalize"].mean(),
        df["speedup_total"].mean(),
        df["speedup_real"].mean(),
    ]
    colors = [C_GPU if s >= 1 else C_CPU for s in speedups]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(ops, speedups, color=colors, zorder=3)
    ax.axvline(1.0, color="gray", linewidth=1.2, linestyle="--", label="Speedup = 1×")

    for bar, val in zip(bars, speedups):
        ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}×", va="center", fontsize=9)

    ax.set_xlabel("Speedup (CPU time / GPU time)")
    ax.set_title("Speedup GPU vs CPU por operación")
    ax.legend()
    save(fig, "speedup_per_operation.png")

# ── Gráfica 3: Desglose GPU — operación vs transferencia ─────────────────────
def plot_transfer_overhead(df):
    avg_op   = ms(df["gpu_total"]).mean()
    avg_xfer = ms(df["gpu_transfer"]).mean()
    avg_cpu  = ms(df["cpu_total"]).mean()

    labels  = ["CPU\n(total)", "GPU\n(operación)", "GPU\n(transferencia\n+ operación)"]
    bottoms = [0, 0, 0]
    ops_h   = [avg_cpu, avg_op, avg_op]
    xfer_h  = [0,       0,      avg_xfer]

    x   = np.arange(3)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x, ops_h,  color=[C_CPU, C_GPU, C_GPU], label="Tiempo de operación", zorder=3)
    ax.bar(x[2:], xfer_h[2:], bottom=ops_h[2:], color=C_XFER,
           label="Transferencia CPU→GPU", zorder=3)

    for i, (o, t) in enumerate(zip(ops_h, xfer_h)):
        total = o + t
        ax.text(i, total + 0.05, f"{total:.2f} ms", ha="center", va="bottom", fontsize=9)

    patch_op   = mpatches.Patch(color=C_GPU,  label="Cómputo GPU")
    patch_xfer = mpatches.Patch(color=C_XFER, label="Transferencia CPU→GPU")
    patch_cpu  = mpatches.Patch(color=C_CPU,  label="CPU")
    ax.legend(handles=[patch_cpu, patch_op, patch_xfer])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Tiempo promedio (ms)")
    ax.set_title("Desglose del tiempo GPU:\ncómputo vs overhead de transferencia")
    save(fig, "transfer_overhead.png")


def main():
    df = load_data()
    print(f"Datos cargados: {len(df)} imágenes\n")
    print("Generando gráficas...")

    plot_bar_times(df)
    plot_speedup_ops(df)
    plot_transfer_overhead(df)

    print(f"\nTodas las gráficas guardadas en: {SCREENSHOTS_DIR}/")


if __name__ == "__main__":
    main()
