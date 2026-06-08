import pandas as pd

cpu = pd.read_csv("results/cpu_results.csv")
gpu = pd.read_csv("results/gpu_results.csv")

df = cpu.merge(gpu, on="image")

df["speedup"] = df["cpu_time"] / df["gpu_time"]

print(df)

print("\nSpeedup promedio:")
print(df["speedup"].mean())

df.to_csv("results/comparison.csv", index=False)