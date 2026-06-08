# Resultados del benchmark CPU vs GPU

Esta carpeta almacena los archivos CSV generados por los scripts de procesamiento y comparación. Son la base de datos del análisis de rendimiento.


## Archivos generados

| Archivo | Generado por | Descripción |
|---|---|---|
| `cpu_results.csv` | `cpu/process_cpu.py` | Tiempos de ejecución en CPU |
| `gpu_results.csv` | `gpu/process_gpu.py` | Tiempos de ejecución en GPU + transferencia |
| `comparison.csv` | `comparison.py` | Tabla unificada con speedups calculados |

---

## Estructura de cada CSV

### `cpu_results.csv`

| Columna | Tipo | Descripción |
|---|---|---|
| `image` | string | Nombre del archivo de imagen |
| `cpu_invert` | float | Tiempo promedio de inversión (segundos) |
| `cpu_gaussian` | float | Tiempo promedio de filtro gaussiano (segundos) |
| `cpu_equalize` | float | Tiempo promedio de ecualización (segundos) |
| `cpu_total` | float | Suma de las tres operaciones (segundos) |

### `gpu_results.csv`


| Columna | Tipo | Descripción |
|---|---|---|
| `gpu_transfer` | float | Tiempo de transferencia CPU→GPU por PCIe (segundos) |
| `gpu_invert` | float | Tiempo de inversión en GPU, sin transferencia (segundos) |
| `gpu_gaussian` | float | Tiempo de gaussiano en GPU, sin transferencia (segundos) |
| `gpu_equalize` | float | Tiempo de ecualización en GPU, sin transferencia (segundos) |
| `gpu_total` | float | Suma de las tres operaciones GPU, sin transferencia (segundos) |

### `comparison.csv`

Contiene todas las columnas anteriores más las métricas de comparación:

| Columna | Descripción |
|---|---|
| `speedup_invert` | `cpu_invert / gpu_invert` |
| `speedup_gaussian` | `cpu_gaussian / gpu_gaussian` |
| `speedup_equalize` | `cpu_equalize / gpu_equalize` |
| `speedup_total` | `cpu_total / gpu_total` (GPU pura) |
| `gpu_total_with_transfer` | `gpu_total + gpu_transfer` |
| `speedup_real` | `cpu_total / gpu_total_with_transfer` (costo real) |

---


### Speedup promedio

| Métrica | Valor |
|---|---|
| Inversión | ~7x |
| Filtro gaussiano | ~21x |
| Ecualización | ~13x |
| Total GPU pura | ~17x |
| **Total GPU + transferencia** | **~12x** |

