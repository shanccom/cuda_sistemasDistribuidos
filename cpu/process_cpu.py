"""
Procesamiento de imágenes médicas - CPU (versión de referencia)
Operaciones: inversión, filtro gaussiano, ecualización de histograma
"""

import os
import time
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter

DATASET_DIR = os.path.join(os.path.dirname(__file__), "../dataset")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../results")
REPETITIONS = 5  # promedio de N ejecuciones para estabilizar la medición


def invert_image(img: np.ndarray) -> np.ndarray:
    """Inversión de píxeles (negativo radiológico)."""
    return np.subtract(255, img, dtype=np.uint8)


def gaussian_blur(img: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Filtro gaussiano para reducción de ruido."""
    return gaussian_filter(img.astype(np.float32), sigma=sigma).astype(np.uint8)


def equalize_histogram(img: np.ndarray) -> np.ndarray:
    """Ecualización de histograma para mejorar contraste."""
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
    cdf = hist.cumsum()
    cdf_min = cdf[cdf > 0].min()
    total_pixels = img.size
    lut = np.round(
        (cdf - cdf_min) / (total_pixels - cdf_min) * 255
    ).astype(np.uint8)
    return lut[img]


def measure(fn, *args, reps=REPETITIONS):
    """Ejecuta fn(*args) `reps` veces y devuelve el tiempo promedio en segundos."""
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        result = fn(*args)
        times.append(time.perf_counter() - start)
    return np.mean(times), result


def main():
    if not os.path.isdir(DATASET_DIR):
        print(f"ERROR: No se encontró la carpeta dataset en '{DATASET_DIR}'")
        return

    images = [
        f for f in os.listdir(DATASET_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not images:
        print("ERROR: No hay imágenes en la carpeta dataset.")
        return

    print(f"Procesando {len(images)} imágenes con CPU ({REPETITIONS} repeticiones cada una)...\n")

    rows = []
    for filename in sorted(images):
        path = os.path.join(DATASET_DIR, filename)
        img = np.array(Image.open(path).convert("L"))

        t_inv,  _ = measure(invert_image,     img)
        t_gaus, _ = measure(gaussian_blur,    img)
        t_hist, _ = measure(equalize_histogram, img)

        h, w = img.shape
        total = t_inv + t_gaus + t_hist

        rows.append({
            "image":       filename,
            "width":       w,
            "height":      h,
            "megapixels":  round(w * h / 1_000_000, 3),
            "cpu_invert":  t_inv,
            "cpu_gaussian": t_gaus,
            "cpu_equalize": t_hist,
            "cpu_total":   total,
        })

        print(
            f"  {filename:<35} | {w}x{h} | "
            f"inv={t_inv*1000:.3f}ms  gauss={t_gaus*1000:.3f}ms  hist={t_hist*1000:.3f}ms  "
            f"total={total*1000:.3f}ms"
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "cpu_results.csv")
    df.to_csv(out, index=False)
    print(f"\nResultados guardados en: {out}")
    print("\nResumen CPU (promedios sobre todas las imágenes):")
    for col in ["cpu_invert", "cpu_gaussian", "cpu_equalize", "cpu_total"]:
        print(f"  {col:<18}: {df[col].mean()*1000:.3f} ms")


if __name__ == "__main__":
    main()
