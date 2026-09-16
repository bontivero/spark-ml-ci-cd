# Clasificador de Flores Iris con PySpark ML, MLflow y CI/CD

[![CI/CD Pipeline](https://github.com/bontivero/spark-ml-ci-cd/actions/workflows/ci.yml/badge.svg)](https://github.com/bontivero/spark-ml-ci-cd/actions/workflows/ci.yml)

## 📖 Descripción

Este proyecto implementa un **clasificador de flores Iris** utilizando **PySpark MLlib** (Regresión Logística) y lo expone a través de una **API REST con FastAPI**. El flujo completo incluye:

- **Preprocesamiento de datos** con PySpark (carga, limpieza, vectorización).
- **Entrenamiento** del modelo con Spark ML.
- **Evaluación** de precisión sobre un conjunto de prueba.
- **Predicción** en tiempo real mediante una API.
- **Gestión de experimentos** con MLflow (métricas, artefactos y versionado de modelos).
- **Tests automáticos** con pytest.
- **Lint con Ruff** para calidad de código.
- **CI/CD con GitHub Actions**: cada push a `main` ejecuta lint y tests, entrena el modelo y publica la imagen Docker en GitHub Container Registry.

El dataset Iris es un clásico en machine learning: contiene 150 muestras de 3 especies de flores (setosa, versicolor, virginica) con 4 características numéricas (longitud/ancho de sépalo y pétalo).

## 🚀 Objetivo

Demostrar un flujo completo de **Machine Learning + Data Engineering + DevOps** usando herramientas modernas:

- **PySpark** para procesamiento distribuido y ML.
- **FastAPI** para servir el modelo como API.
- **GitHub Actions** para integración y entrega continua, con publicación de imágenes en GitHub Container Registry.
- **MLflow** para trazabilidad de experimentos y modelos.

## Evaluación del modelo

El modelo se evalúa sobre un conjunto de prueba (20% de los datos) y se calculan las siguientes métricas:

- **Accuracy**: porcentaje de predicciones correctas.
- **Precisión ponderada (Weighted Precision)**: media de la precisión de cada clase ponderada por el número de muestras.
- **Recall ponderado (Weighted Recall)**: media del recall de cada clase ponderada.
- **F1-score ponderado**: media armónica de precisión y recall.

Además, se genera una **matriz de confusión** que muestra cuántas muestras de cada clase real fueron predichas en cada clase. La diagonal principal representa los aciertos; los elementos fuera de ella son errores.

En MLflow se guardan las métricas, la matriz y el modelo entrenado para su trazabilidad.

## 📁 Estructura del proyecto
```text
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                 # Pipeline CI/CD
│   │   └── codeql.yml             # Análisis de seguridad
│   └── dependabot.yml             # Actualización automática de dependencias
├── app/
│   └── main.py                    # API FastAPI
├── data/
│   ├── iris.csv                   # Dataset completo (150 filas)
│   └── test_data.csv              # Datos de prueba para validación
├── src/
│   ├── data_preprocessing.py      # Carga y transformación de datos
│   ├── train_model.py             # Entrenamiento del modelo
│   └── predict.py                 # Funciones de predicción
├── tests/
│   ├── test_preprocessing.py
│   ├── test_training.py
│   └── test_prediction.py
├── Dockerfile                     # Imagen Docker de la API
├── .dockerignore                  # Archivos excluidos del build
├── pyproject.toml                 # Configuración de Ruff
├── run_experiments.py             # Ejecutor de múltiples experimentos
├── check_mlflow.py                # Utilidad para inspeccionar MLflow
├── requirements.txt
└── README.md
```

## ⚙️ Requisitos

- Python 3.10+
- Java 8/11/17 (necesario para PySpark)
- Dependencias de Python (ver `requirements.txt`)

## 🛠️ Instalación y uso local

1. **Clonar el repositorio**

   ```bash
   git clone https://github.com/tu-usuario/spark-ml-ci-cd-sin-docker.git
   cd spark-ml-ci-cd-sin-docker
   ```
2. **Crear entorno virtual e instalar dependencias**

   ```bash
   python -m venv venv
   source venv/bin/activate   # En Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Entrenar el modelo**

  ```bash
  python -m src.train_model
  ```
  Esto generará la carpeta models/ con el modelo entrenado.

4. **Ejecutar la API**

  ```bash
  uvicorn app.main:app --reload
  ```

5. **Probar la API**

   Abre http://localhost:8000/docs para ver la documentación interactiva o usa curl:
   ```bash
   curl -X POST http://localhost:8000/predict \
   -H "Content-Type: application/json" \
   -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
   ```

   Respuesta esperada:
   ```json
   {"prediction":0,"species":"setosa"}
   ```

## 🧪 Tests

Ejecuta todos los tests con:
  ```bash
  pytest tests/
  ```

Los tests validan:
  - Carga correcta del dataset (150 filas).
  - Transformaciones del pipeline (features, indexed_label).
  - Precisión del modelo >= 90%.
  - Predicciones correctas sobre datos de prueba (data/test_data.csv).

## 📊 Gestión de experimentos con MLflow

Los experimentos se registran en una base de datos SQLite local (`mlflow.db`) bajo el experimento `iris_classification`. Para cada run se guardan:

- **Parámetros**: tipo de modelo, hiperparámetros (`maxIter`, `regParam`, `numTrees`, `maxDepth`).
- **Métricas**: accuracy, weighted precision, weighted recall, F1 y métricas por clase.
- **Artefactos**: matriz de confusión (`confusion_matrix.npy`) y el modelo entrenado.

### Ejecutar múltiples experimentos

El script `run_experiments.py` lanza varias configuraciones de los tres algoritmos:

```bash
python run_experiments.py
```

Visualizar resultados
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## 🔄 CI/CD con GitHub Actions

El pipeline definido en [`.github/workflows/ci.yml`](.github/workflows/ci.yml) se ejecuta automáticamente y consta de dos jobs: **test** y **build-and-push**.

### Características del pipeline

- **Control de concurrencia**: si se hace un nuevo push a la misma rama mientras un workflow está en curso, el anterior se cancela automáticamente.
- **Timeouts**: cada job tiene un límite de tiempo (15 min para tests, 30 min para build) para evitar ejecuciones colgadas.
- **Cache de pip**: las dependencias de Python se cachean entre ejecuciones para acelerar la instalación.
- **Lint con Ruff**: se verifica la calidad y el estilo del código antes de ejecutar los tests.

### Job `test`

Se ejecuta en cada **push a `main`** y en cada **pull request** hacia `main`:

1. Configura Java 11 y Python 3.12.
2. Instala dependencias con cache de pip.
3. Ejecuta **Ruff** para verificar la calidad del código.
4. Ejecuta los **tests** con pytest.

### Job `build-and-push`

Se ejecuta **solo en push a `main`** y **solo si el job `test` pasa**:

1. Configura Java 11 y Python 3.12 (con cache de pip).
2. Entrena el modelo (`python -m src.train_model`), generando la carpeta `models/`.
3. Configura Docker Buildx.
4. Hace login en **GitHub Container Registry (GHCR)** usando `docker/login-action`.
5. Extrae los tags y labels con `docker/metadata-action`.
6. Construye y sube la imagen Docker con `docker/build-push-action`, usando cache de capas con GitHub Actions.

### Imágenes publicadas

Las imágenes se publican en GHCR con las siguientes etiquetas:

- `ghcr.io/tu-usuario/spark-ml-ci-cd:latest` → última build de `main`.
- `ghcr.io/tu-usuario/spark-ml-ci-cd:main` → build de la rama `main`.
- `ghcr.io/tu-usuario/spark-ml-ci-cd:sha-<commit>` → build asociada a un commit específico.

Puedes descargar y ejecutar la imagen localmente:

```bash
docker pull ghcr.io/tu-usuario/spark-ml-ci-cd:latest
docker run -p 8000:8000 ghcr.io/tu-usuario/spark-ml-ci-cd:latest
```

## 🧠 Tecnologías utilizadas

- **Apache Spark / PySpark** — procesamiento distribuido y MLlib.
- **MLflow** — tracking de experimentos y versionado de modelos.
- **FastAPI** — API REST para servir el modelo.
- **Uvicorn** — servidor ASGI.
- **pytest** — tests automatizados.
- **Ruff** — linting y calidad de código.
- **Docker** — contenedor de la API.
- **GitHub Actions** — CI/CD, CodeQL y Dependabot.
- **GitHub Container Registry (GHCR)** — registro de imágenes Docker.
- **Python 3.12** — versión usada en CI/CD.

## 📄 Licencia

Este proyecto es de uso educativo y libre.
