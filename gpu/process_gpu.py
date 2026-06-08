import os
import time
import numpy as np
import cupy as cp
import pandas as pd
from PIL import Image

DATASET_DIR = "../dataset"

results = []

for filename in os.listdir(DATASET_DIR):

    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    path = os.path.join(DATASET_DIR, filename)

    image = Image.open(path).convert("L")

    img = np.array(image)

    gpu_img = cp.asarray(img)

    cp.cuda.Stream.null.synchronize()

    start = time.perf_counter()

    gpu_output = 255 - gpu_img

    cp.cuda.Stream.null.synchronize()

    elapsed = time.perf_counter() - start

    results.append({
        "image": filename,
        "gpu_time": elapsed
    })

    print(f"{filename} -> {elapsed:.8f}s")

df = pd.DataFrame(results)

os.makedirs("../results", exist_ok=True)

df.to_csv("../results/gpu_results.csv", index=False)

print("GPU terminado")