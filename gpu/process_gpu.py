"""
Procesamiento de imágenes médicas - GPU
Operaciones: inversión, filtro gaussiano, ecualización de histograma
"""

import os
import time
import numpy as np
import pandas as pd
from PIL import Image

try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter as cp_gaussian_filter
except ImportError:
    print("ERROR: CuPy no está instalado.")
    print("Instalación: pip install cupy-cuda12x   (ajusta según tu versión de CUDA)")
    raise

DATASET_DIR = os.path.join(os.path.dirname(__file__), "../dataset")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../results")
REPETITIONS = 5


# Kernels GPU

def gpu_invert(gpu_img: cp.ndarray) -> cp.ndarray:
    """Inversión de píxeles sobre GPU."""
    return cp.subtract(255, gpu_img, dtype=cp.uint8)


def gpu_gaussian(gpu_img: cp.ndarray, sigma: float = 2.0) -> cp.ndarray:
    """Filtro gaussiano sobre GPU."""
    return cp_gaussian_filter(gpu_img.astype(cp.float32), sigma=sigma).astype(cp.uint8)


def gpu_equalize(gpu_img: cp.ndarray) -> cp.ndarray:
    """Ecualización de histograma sobre GPU."""
    hist = cp.bincount(gpu_img.ravel(), minlength=256).astype(cp.float64)
    cdf  = cp.cumsum(hist)
    cdf_min     = float(cdf[cdf > 0][0])
    total_pixels = int(gpu_img.size)
    lut  = cp.round((cdf - cdf_min) / (total_pixels - cdf_min) * 255).astype(cp.uint8)
    return lut[gpu_img]


def sync():
    """Barrera de sincronización CUDA para medición correcta."""
    cp.cuda.Stream.null.synchronize()


def measure_gpu(fn, gpu_img, reps=REPETITIONS):
    """
    Mide el tiempo de la operación GPU pura (sin transferencia de memoria).
    Retorna (tiempo_promedio_segundos, último_resultado).
    """
    times = []
    result = None
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        result = fn(gpu_img)
        sync()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), result


def warmup():
    """
    Calentamiento del contexto CUDA.
    Sin esto, la primera operación real incluye la latencia de inicialización del driver y contamina las métricas.
    """
    print("Calentando contexto CUDA...", end=" ", flush=True)
    dummy = cp.zeros((512, 512), dtype=cp.uint8)
    for _ in range(3):
        _ = cp.subtract(255, dummy)
        sync()
    del dummy
    print("listo.\n")


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

    # Información del dispositivo
    device = cp.cuda.Device(0)
    props  = cp.cuda.runtime.getDeviceProperties(device.id)
    gpu_name = props["name"].decode()
    print(f"GPU detectada: {gpu_name}")
    print(f"Procesando {len(images)} imágenes ({REPETITIONS} repeticiones cada una)...\n")

    warmup()

    rows = []
    for filename in sorted(images):
        path = os.path.join(DATASET_DIR, filename)
        img  = np.array(Image.open(path).convert("L"))

        # Transferencia CPU → GPU (medida aparte, informativa)
        sync()
        t_transfer_start = time.perf_counter()
        gpu_img = cp.asarray(img)
        sync()
        t_transfer = time.perf_counter() - t_transfer_start

        # Operaciones GPU puras
        t_inv,  _ = measure_gpu(gpu_invert,   gpu_img)
        t_gaus, _ = measure_gpu(gpu_gaussian,  gpu_img)
        t_hist, _ = measure_gpu(gpu_equalize,  gpu_img)

        del gpu_img

        h, w = img.shape
        total = t_inv + t_gaus + t_hist

        rows.append({
            "image":         filename,
            "width":         w,
            "height":        h,
            "megapixels":    round(w * h / 1_000_000, 3),
            "gpu_transfer":  t_transfer,
            "gpu_invert":    t_inv,
            "gpu_gaussian":  t_gaus,
            "gpu_equalize":  t_hist,
            "gpu_total":     total,
        })

        print(
            f"  {filename:<35} | {w}x{h} | "
            f"xfer={t_transfer*1000:.3f}ms  "
            f"inv={t_inv*1000:.3f}ms  gauss={t_gaus*1000:.3f}ms  hist={t_hist*1000:.3f}ms  "
            f"total={total*1000:.3f}ms"
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df  = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "gpu_results.csv")
    df.to_csv(out, index=False)
    print(f"\nResultados guardados en: {out}")
    print("\nResumen GPU (promedios sobre todas las imágenes):")
    for col in ["gpu_transfer", "gpu_invert", "gpu_gaussian", "gpu_equalize", "gpu_total"]:
        print(f"  {col:<20}: {df[col].mean()*1000:.3f} ms")


if __name__ == "__main__":
    main()
