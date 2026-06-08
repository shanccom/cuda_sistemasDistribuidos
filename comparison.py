"""
Comparación CPU vs GPU: calcula speedup por operación y por imagen.
"""

import os
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def main():
    cpu_path = os.path.join(RESULTS_DIR, "cpu_results.csv")
    gpu_path = os.path.join(RESULTS_DIR, "gpu_results.csv")

    if not os.path.isfile(cpu_path):
        print(f"ERROR: Falta '{cpu_path}'. Ejecutar primero cpu/process_cpu.py")
        return
    if not os.path.isfile(gpu_path):
        print(f"ERROR: Falta '{gpu_path}'. Ejecutar primero gpu/process_gpu.py")
        return

    cpu = pd.read_csv(cpu_path)
    gpu = pd.read_csv(gpu_path)

    df = cpu.merge(gpu, on=["image", "width", "height", "megapixels"])

    # Speedup por operación (sin transferencia)
    df["speedup_invert"]   = df["cpu_invert"]   / df["gpu_invert"]
    df["speedup_gaussian"] = df["cpu_gaussian"]  / df["gpu_gaussian"]
    df["speedup_equalize"] = df["cpu_equalize"]  / df["gpu_equalize"]
    df["speedup_total"]    = df["cpu_total"]     / df["gpu_total"]

    # Speedup incluyendo transferencia (coste real)
    df["gpu_total_with_transfer"] = df["gpu_total"] + df["gpu_transfer"]
    df["speedup_real"]            = df["cpu_total"] / df["gpu_total_with_transfer"]

    out = os.path.join(RESULTS_DIR, "comparison.csv")
    df.to_csv(out, index=False)

    print("\n" + "═" * 70)
    print("PROMEDIOS DE SPEEDUP")
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
