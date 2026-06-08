"""
Comparación CPU vs GPU: calcula speedup por operación y por imagen.
Ejecutar desde la raíz del proyecto después de process_cpu.py y process_gpu.py.
"""

import os
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def main():
    cpu_path = os.path.join(RESULTS_DIR, "cpu_results.csv")
    gpu_path = os.path.join(RESULTS_DIR, "gpu_results.csv")

    if not os.path.isfile(cpu_path):
        print(f"ERROR: Falta '{cpu_path}'. Ejecuta primero cpu/process_cpu.py")
        return
    if not os.path.isfile(gpu_path):
        print(f"ERROR: Falta '{gpu_path}'. Ejecuta primero gpu/process_gpu.py")
        return

    cpu = pd.read_csv(cpu_path)
    gpu = pd.read_csv(gpu_path)

    df = cpu.merge(gpu, on=["image", "width", "height", "megapixels"])

    # Speedup por operación (GPU pura, sin transferencia)
    df["speedup_invert"]   = df["cpu_invert"]   / df["gpu_invert"]
    df["speedup_gaussian"] = df["cpu_gaussian"]  / df["gpu_gaussian"]
    df["speedup_equalize"] = df["cpu_equalize"]  / df["gpu_equalize"]
    df["speedup_total"]    = df["cpu_total"]     / df["gpu_total"]

    # Speedup incluyendo transferencia (coste real si hay un solo procesamiento)
    df["gpu_total_with_transfer"] = df["gpu_total"] + df["gpu_transfer"]
    df["speedup_real"]            = df["cpu_total"] / df["gpu_total_with_transfer"]

    out = os.path.join(RESULTS_DIR, "comparison.csv")
    df.to_csv(out, index=False)

    # ── Reporte en consola ────────────────────────────────────────────────────
    cols_display = [
        "image", "megapixels",
        "cpu_total", "gpu_total", "gpu_total_with_transfer",
        "speedup_total", "speedup_real",
        "speedup_invert", "speedup_gaussian", "speedup_equalize",
    ]
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df[cols_display].to_string(index=False))

    print("\n" + "═" * 70)
    print("RESUMEN DE SPEEDUP (promedio sobre todas las imágenes)")
    print("═" * 70)
    for key, label in [
        ("speedup_invert",   "Inversión          "),
        ("speedup_gaussian", "Filtro gaussiano   "),
        ("speedup_equalize", "Ecualización hist. "),
        ("speedup_total",    "TOTAL (GPU pura)   "),
        ("speedup_real",     "TOTAL (GPU + xfer) "),
    ]:
        mean = df[key].mean()
        bar  = "█" * min(int(mean), 50)
        print(f"  {label}: {mean:6.2f}x  {bar}")

    print(f"\nArchivo de comparación guardado en: {out}")


if __name__ == "__main__":
    main()
