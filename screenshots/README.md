# Evidencias de ejecución y gráficas de análisis

Esta carpeta almacena las **3 gráficas de análisis de rendimiento** generadas por `graph_results.py`, así como capturas de pantalla de la ejecución de los scripts.

## Archivos generados por `graph_results.py`

| Archivo | Descripción |
|---|---|
| `bar_times_per_operation.png` | **Tiempos promedio** CPU vs GPU por operación |
| `speedup_per_operation.png` | Factor de **aceleración** GPU vs CPU por operación (barras horizontales) |
| `transfer_overhead.png` | Tiempo GPU: cómputo puro vs overhead de transferencia |


## Descripción de cada gráfica

### 1. `bar_times_per_operation.png` — Tiempos por operación

Gráfica de barras que compara el **tiempo promedio en milisegundos** de CPU (rojo) y GPU (azul) para cada una de las tres operaciones y el total.


### 2. `speedup_per_operation.png` — Speedup por operación

Gráfica de barras que muestra el **factor de aceleración** (speedup = tiempo CPU / tiempo GPU) para cada operación.

- La línea vertical punteada en `1.0` marca el punto de equilibrio.
- Barras a la derecha de `1.0` = GPU es más rápida.

El filtro gaussiano obtiene el mayor speedup (~21x), seguido de la ecualización (~13x) y la inversión (~7x).

---

### 3. `transfer_overhead.png` — Overhead de transferencia

Gráfica de barras apiladas que compara tres escenarios:

| Barra | Descripción |
|---|---|
| **CPU (total)** | Costo total en CPU |
| **GPU (operación)** | Solo el cómputo en GPU, sin mover datos |
| **GPU (transferencia + operación)** | Costo real: mover datos por PCIe + cómputo |


El overhead de transferencia representa ~30% del tiempo GPU total, pero aun con ese costo, la GPU es ~12x más rápida que la CPU.

---

## Generar las gráficas

```bash
python graph_results.py
```


## Script que genera estas gráficas

- `graph_results.py`  — requiere `results/comparison.csv`