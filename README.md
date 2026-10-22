# Evaluación A/B de Pooling Morfológico Adaptativo (RGB) vs. Max Pooling Estándar para Detección de Melanoma

**Trabajo Final de Grado / Tesis de Ingeniería**  
*Clasificación Asistida por Computadora (CAD) de Lesiones Dermatoscópicas en Imágenes Médicas*

---

## 1. Descripción del Proyecto

Este repositorio implementa un banco de pruebas experimental **A/B riguroso** para evaluar el impacto de sustituir el submuestreo clásico (*Max Pooling*) por una capa personalizada de **Pooling Morfológico Adaptativo Vectorial** (`AdaptiveMorphPooling2d`) sobre imágenes dermatoscópicas en color.

El problema se formula como una **clasificación binaria**:
* **Clase 0 (`benign`):** Lesión benigna (Paciente sano).
* **Clase 1 (`malignant`):** Melanoma maligno (Paciente enfermo).

### Fundamentación Teórica y Planteamiento del Problema
* **Limitación del Max Pooling estándar:** En imágenes RGB de 3 canales, `nn.MaxPool2d` realiza un *pooling marginal* (toma el valor máximo en R, G y B de forma completamente independiente). Esto destruye la correlación espectral y genera **colores espurios o falsos**, degradando microestructuras patológicas esenciales como el velo azul-blanco, la red pigmentaria atípica y glóbulos periféricos.
* **Solución propuesta (Morph Pooling Vectorial):** La capa `AdaptiveMorphPooling2d` aplica un **ordenamiento vectorial reducido ($h$-ordering)**. Calcula la dispersión/varianza local por canal mediante el descriptor **SMO** (*Statistical Measure of Smoothness*), construye un vector de pesos adaptativo $\psi = [\psi_R, \psi_G, \psi_B]$ y selecciona el **píxel físico real intacto** con mayor puntuación, preservando la fidelidad cromática y de bordes.

---

## 2. Dataset y Partición de Datos (Train / Test)

### 2.1 Origen del Dataset
El experimento utiliza el conjunto de datos dermatoscópicos curado a partir del archivo ISIC (*International Skin Imaging Collaboration*), disponible públicamente en Kaggle:
* **Enlace oficial de descarga:** [Skin Cancer: Malignant vs. Benign (Kaggle)](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign?resource=download)

### 2.2 Estructura y Distribución de las Particiones
El dataset contiene un total de **3,297 imágenes dermatoscópicas** en formato JPEG, pre-particionadas en una división estándar 80/20:

| Partición | Directorio | Benignos (Sanos / Clase 0) | Malignos (Melanoma / Clase 1) | Total Imágenes | Porcentaje |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Entrenamiento (Train)** | `./archive/train/` | 1,440 (54.6%) | 1,197 (45.4%) | **2,637** | ~80.0% |
| **Evaluación (Test Ciego)** | `./archive/test/` | 360 (54.5%) | 300 (45.5%) | **660** | ~20.0% |
| **Total Global** | | **1,800** | **1,497** | **3,297** | **100.0%** |

> **Nota sobre el balance de clases:** Tanto el conjunto de entrenamiento como el de prueba mantienen una distribución prácticamente idéntica (~54.5% benignos vs. ~45.5% malignos). Este equilibrio natural permite evaluar métricas como Sensibilidad, Especificidad y Exactitud sin el sesgo patológico habitual de datasets altamente desbalanceados.

```
archive/
├── train/
│   ├── benign/       (1,440 imágenes)
│   └── malignant/    (1,197 imágenes)
└── test/
    ├── benign/       (360 imágenes)
    └── malignant/    (300 imágenes)
```

---

### 2.3 Preprocesamiento y Flujo de Ingesta a la Red Neuronal

Para alimentar eficientemente los datos a la red neuronal convolucional, el pipeline implementado en [train_eval.py](train_eval.py) sigue las siguientes etapas:

#### 1. Asignación de Etiquetas (Ground Truth)
Se emplea `torchvision.datasets.ImageFolder`, el cual asigna automáticamente los identificadores numéricos en orden alfabético de carpetas:
* `archive/.../benign` $\rightarrow$ Etiqueta `0.0` (Clase negativa / Sano).
* `archive/.../malignant` $\rightarrow$ Etiqueta `1.0` (Clase positiva / Enfermo).

#### 2. Pipeline de Transformaciones (`transform`)
Cada imagen se procesa mediante las transformaciones de `torchvision.transforms`:
1. `transforms.Resize((112, 112))`: Estandariza la resolución espacial a $112 \times 112$ píxeles. Esta resolución optimiza significativamente el cómputo de la operación de extracción de ventanas (`F.unfold`) y agiliza el tiempo de iteración por época manteniendo los rasgos clínicos macroscópicos de la lesión.
2. `transforms.ToTensor()`: Convierte la imagen PIL a un tensor de PyTorch de dimensiones `[3, 112, 112]` y normaliza los valores de intensidad desde el rango entero $[0, 255]$ al rango de punto flotante $[0.0, 1.0]$.

#### 3. Precarga Acelerada en Memoria RAM (`FastRAMDataset`)
En lugar de cargar y decodificar miles de archivos JPEG individuales desde el disco rígido/SSD en cada época (lo que suele provocar cuellos de botella severos de I/O en Windows), se diseñó la clase personalizada `FastRAMDataset`:
* Durante la inicialización (toma ~15 segundos), lee, transforma y apila (`torch.stack`) **todas las imágenes directamente en la memoria RAM**.
* Las etiquetas se almacenan como tensores de punto flotante `torch.float32`.
* Durante el bucle de entrenamiento, el acceso a cada imagen es inmediato (latencia en memoria de microsegundos), aumentando la velocidad entre $4\times$ y $10\times$ respecto a la lectura convencional de disco.

#### 4. Generación de Lotes y Paridad Experimental (`DataLoader`)
* **Entrenamiento (`train_loader`):** Agrupa las imágenes en minilotes de tamaño configurable (`batch_size=64` por defecto) y activa el barajado estocástico (`shuffle=True`).
* **Prueba (`test_loader`):** Evalúa con `shuffle=False` para una prueba ciega determinista.
* **Reproducibilidad estricta (*Ceteris Paribus*):** Al llamar a la función `set_seed(42)` inmediatamente antes de entrenar tanto el Modelo Baseline como el Modelo Propuesto, PyTorch garantiza que **ambos modelos reciban exactamente la misma secuencia de lotes barajados en cada época**, asegurando que cualquier diferencia en el desempeño se deba únicamente a la capa de pooling morfológico.

---

## 3. Arquitectura de los Módulos

El proyecto está diseñado bajo una arquitectura modular y desacoplada:

```
21-10/
├── archive/                      # Dataset ISIC organizado en train/ y test/ (benign/malignant)
│   ├── train/
│   └── test/
├── morph_pool_layer.py           # Capa PyTorch AdaptiveMorphPooling2d con vector de pesos psi
├── models.py                     # Red neuronal convolucional CNNClassifier (Red A vs Red B)
├── train_eval.py                 # Orquestador del entrenamiento A/B con soporte de argumentos CLI
├── report_generator.py           # Motor generador del informe científico interactivo en HTML con SVGs
├── reporte_experimento_melanoma.html # Informe generado con métricas, curvas y matrices
└── README.md                     # Documentación completa y guía de interpretación
```

### Detalle de Componentes Clave:
1. **[morph_pool_layer.py](morph_pool_layer.py):**
   * Extrae ventanas $2 \times 2$ con `F.unfold`.
   * Calcula el descriptor $\text{SMO} = 1 - \frac{1}{1 + \text{Var}}$.
   * **Manejo de varianza cero:** Si una ventana es plana ($Var \approx 0$, ej. fondo negro del dermatoscopio), asigna pesos equitativos $\psi = [1/3, 1/3, 1/3]$, seleccionando por luminosidad media y eliminando sesgos numéricos.
   * Selecciona el vector de color intacto mediante `torch.gather`.
2. **[models.py](models.py):**
   * **Red A (Baseline):** Aplica `nn.MaxPool2d(2, 2)` sobre la imagen de 3 canales RGB antes de la extracción de características.
   * **Red B (Propuesto):** Aplica `AdaptiveMorphPooling2d(2, 2)` sobre los 3 canales RGB.
   * Ambas redes comparten idénticos bloques convolucionales (`Conv2d`, `BatchNorm2d`, `ReLU`, `MaxPool2d`) y clasificador denso final.
3. **[train_eval.py](train_eval.py):**
   * Implementa `FastRAMDataset` para precargar las imágenes en tensores en memoria RAM.
   * Inicializa la semilla determinista (`set_seed(42)`) al inicio de cada experimento (*ceteris paribus*).
   * Dispone de argumentos por línea de comandos (`argparse`) para controlar el número de épocas y otros parámetros.
4. **[report_generator.py](report_generator.py):**
   * Construye un informe HTML con gráficos vectoriales SVG (curvas de pérdida por época y curvas ROC), matrices de confusión clínicas con heatmaps y tablas científicas formales.

---

## 4. Instalación y Requisitos

### Requisitos de Software:
* Python 3.10 o superior.
* Entorno virtual de Python (`venv`).

### Dependencias Principales:
```bash
pip install torch torchvision scikit-learn numpy
```

---

## 5. Guía de Ejecución por Terminal (Argumentos CLI)

El script [train_eval.py](train_eval.py) incluye un sistema completo de argumentos por línea de comandos (`argparse`) que permite configurar el número de **épocas (etapas)**, tamaño de lote, tasa de aprendizaje y nombre del informe.

### Ver la Ayuda de Todos los Comandos Disponibles:
```bash
python train_eval.py --help
```

Salida esperada:
```text
usage: train_eval.py [-h] [-e EPOCHS] [-b BATCH_SIZE] [--lr LR]
                     [--metric {SMO,VAR}] [-d DATA_DIR] [-o OUTPUT]

Experimento A/B: Pooling Morfologico Adaptativo vs MaxPool2d en Imagenes Dermatoscopicas (Melanoma)

options:
  -h, --help            Muestra este mensaje de ayuda y termina
  -e, --epochs EPOCHS   Numero de etapas/epocas de entrenamiento por modelo (default: 5)
  -b, --batch-size BATCH_SIZE
                        Tamano del lote / batch size (default: 64)
  --lr LR               Tasa de aprendizaje / Learning rate (default: 0.001)
  --metric {SMO,VAR}    Metrica estadistica para el descriptor morfologico (default: SMO)
  -d, --data-dir DIR    Ruta al directorio de datos que contiene 'train' y 'test' (default: ./archive)
  -o, --output OUTPUT   Ruta/nombre del informe HTML de salida (default: reporte_experimento_melanoma.html)
```

### Comandos de Ejemplo:

#### A. Ejecución Rápida por Defecto (5 Épocas):
```bash
python train_eval.py
```

#### B. Especificar el Número de Épocas / Etapas de Prueba:
Para probar con **10 épocas**:
```bash
python train_eval.py --epochs 10
```
O usando la forma abreviada (`-e`):
```bash
python train_eval.py -e 15
```

#### C. Personalizar Múltiples Hiperparámetros:
Por ejemplo: 10 épocas, batch size de 32, learning rate de 0.0005 y guardando en un archivo HTML específico:
```bash
python train_eval.py -e 10 -b 32 --lr 0.0005 -o reporte_final_10epocas.html
```

---

## 6. Guía de Interpretación de Resultados

Al finalizar el entrenamiento, el sistema imprime un resumen en la terminal y genera el informe interactivo en **HTML** ([reporte_experimento_melanoma.html](reporte_experimento_melanoma.html)), el cual puede abrirse en cualquier navegador web o imprimirse en PDF.

A continuación se explica el significado clínico y estadístico de cada indicador:

### 1. Sensibilidad / Recall ($\text{TPR}$) — *La Métrica Reina en Oncología*
$$\text{Sensibilidad} = \frac{TP}{TP + FN}$$
* **Significado clínico:** Proporción de melanomas reales que el modelo detectó correctamente.
* **Interpretación:** En cáncer de piel, un **Falso Negativo ($FN$)** representa un paciente con cáncer al que se le dice erróneamente que está sano, perdiendo una ventana de intervención quirúrgica temprana crucial. Por ello, **un aumento en la Sensibilidad representa la ganancia clínica más valiosa de la tesis**.

### 2. Especificidad ($\text{TNR}$) — *Control de Biopsias Innecesarias*
$$\text{Especificidad} = \frac{TN}{TN + FP}$$
* **Significado clínico:** Proporción de lesiones benignas correctamente clasificadas como benignas.
* **Interpretación:** Un **Falso Positivo ($FP$)** somete a un paciente sano a estrés psicológico y a una biopsia diagnóstica invasiva evitable. Una alta especificidad evita la sobrecarga asistencial del sistema de salud.

### 3. Área Bajo la Curva ROC (AUC-ROC)
* **Significado:** Mide la capacidad global del modelo para ordenar las predicciones asignando mayores probabilidades de malignidad a los melanomas que a los lunares sanos, **independientemente del umbral de decisión** (0.5).
* **Interpretación:**
  * $\text{AUC} = 0.5$: Clasificador aleatorio (equivalente a lanzar una moneda).
  * $\text{AUC} \ge 0.85$: Buen rendimiento clínico.
  * $\text{AUC} \ge 0.90$: Rendimiento excelente de grado diagnóstico.

### 4. Exactitud (Accuracy)
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
* **Significado:** Porcentaje total de decisiones correctas.
* **Interpretación:** Aunque es la métrica más popular, en medicina no debe evaluarse de forma aislada, pues no distingue entre el coste de un falso positivo vs. un falso negativo.

---

### 5. Lectura de las Matrices de Confusión Cuadrante a Cuadrante

El informe presenta dos matrices $2 \times 2$ (una para el Baseline y otra para el Propuesto):

```
                        PREDICCIÓN DEL MODELO
                     Benigno (0)       Maligno (1)
REALIDAD  Benigno (0) [  TN  ]        [  FP  ]   --> Alarma falsa (Biopsia evitable)
          Maligno (1) [  FN  ]        [  TP  ]   --> Detección exitosa
                         ▲
                         │
                 ¡RIESGO CLÍNICO MÁXIMO!
               (Melanoma no detectado)
```

* **Verdadero Negativo ($TN$):** Lunar benigno diagnosticado como sano.
* **Falso Positivo ($FP$):** Lunar benigno clasificado como maligno.
* **Falso Negativo ($FN$):** **Melanoma real clasificado como benigno**. Si la Red B reduce el valor de $FN$ respecto a la Red A, la hipótesis queda empíricamente respaldada.
* **Verdadero Positivo ($TP$):** Melanoma correctamente detectado para tratamiento inmediato.

---

### 6. Interpretación de los Gráficos Vectoriales (SVG)

1. **Curva de Evolución de la Pérdida (Loss Curve):**
   * Permite verificar la velocidad de convergencia en cada época.
   * Una curva que desciende de forma suave y sostenida indica estabilidad en el entrenamiento y ausencia de divergencia del gradiente.
2. **Curva ROC ($FPR$ vs $TPR$):**
   * Representa el compromiso entre Sensibilidad y Falsas Alarmas.
   * La curva que más se acerque a la esquina superior izquierda ($FPR=0, TPR=1$) corresponde al mejor modelo clasificador.

---

*Proyecto desarrollado con PyTorch, Torchvision y Scikit-Learn.*
