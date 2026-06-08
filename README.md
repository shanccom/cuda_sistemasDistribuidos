#  Procesamiento Masivo de Imágenes Médicas con CUDA

> **Grupo 3 — Sistemas Distribuidos**  
> Tecnología asignada: **CUDA** | Caso de aplicación: Procesamiento masivo de imágenes médicas mediante GPU



##  Descripción general

Se implementa y compara el procesamiento paralelo de imágenes médicas (radiografías de tórax) usando dos enfoques:

- **CPU (secuencial):** procesamiento con NumPy y SciPy como línea base de referencia.
- **GPU (CUDA):** procesamiento masivamente paralelo con CuPy, aprovechando los miles de núcleos de la GPU.

Se aplican tres operaciones de procesamiento de imagen sobre un dataset:

| Operación | Descripción clínica |
|---|---|
| **Inversión (negativo)** | Negativo radiológico para facilitar la lectura de densidades |
| **Filtro gaussiano** | Reducción de ruido antes de segmentación |
| **Ecualización de histograma** | Mejora de contraste para detección de anomalías |

---

## } Arquitectura del proyecto

```
cuda_sistemasDistribuidos/
│
├── cpu/
│   ├── process_cpu.py        # Pipeline CPU (NumPy + SciPy)
│   └── README.md
│
├── gpu/
│   ├── process_gpu.py        # Pipeline GPU (CuPy / CUDA)
│   └── README.md
│
├── dataset/
│   ├── *.png                 # Radiografías de tórax (NIH ChestX-ray)
│   └── README.md
│
├── results/
│   ├── cpu_results.csv       # Tiempos por imagen y operación (CPU)
│   ├── gpu_results.csv       # Tiempos por imagen y operación (GPU)
│   ├── comparison.csv        # Tabla unificada con speedups
│   └── README.md
│
├── screenshots/
│   ├── bar_times_per_operation.png   # Gráfica: tiempos CPU vs GPU
│   ├── speedup_per_operation.png     # Gráfica: speedup por operación
│   ├── transfer_overhead.png         # Gráfica: overhead de transferencia
│   └── README.md
│
├── comparison.py             # Calcula y reporta speedups CPU vs GPU
├── graph_results.py          # Genera las tres gráficas de análisis
├── requirements.txt          # Dependencias del proyecto
└── README.md                 ← este archivo
```

---

## Requisitos del sistema

| Componente | Versión mínima |
|---|---|
| Python | 3.9+ |
| CUDA Toolkit | 12.x |
| GPU NVIDIA | Compute Capability 6.0+ |
| RAM GPU | 4 GB recomendado |
| RAM CPU | 8 GB recomendado |

---

##  Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/<tu-usuario>/cuda_sistemasDistribuidos.git
cd cuda_sistemasDistribuidos
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv .venv
source .venv/bin/activate       
.venv\Scripts\activate       
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

> **Nota:** `cupy-cuda12x` requiere CUDA 12.x instalado en el sistema.  
> Si usas CUDA 11.x, reemplaza en `requirements.txt` por `cupy-cuda11x`.

---

##  Ejecución paso a paso

Los scripts deben ejecutarse en este orden desde la raíz del proyecto:

```bash
# Paso 1: Procesamiento CPU (genera results/cpu_results.csv)
python cpu/process_cpu.py

# Paso 2: Procesamiento GPU (genera results/gpu_results.csv)
python gpu/process_gpu.py

# Paso 3: Comparar resultados (genera results/comparison.csv)
python comparison.py

# Paso 4: Generar gráficas (genera screenshots/*.png)
python graph_results.py
```
