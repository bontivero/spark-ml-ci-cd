# Clasificador de Flores Iris con PySpark ML y CI/CD

![CI/CD Pipeline](https://github.com/tu-usuario/spark-ml-ci-cd-sin-docker/actions/workflows/ci.yml/badge.svg)

## 📖 Descripción

Este proyecto implementa un **clasificador de flores Iris** utilizando **PySpark MLlib** (Regresión Logística) y lo expone a través de una **API REST con FastAPI**. El flujo completo incluye:

- **Preprocesamiento de datos** con PySpark (carga, limpieza, vectorización).
- **Entrenamiento** del modelo con Spark ML.
- **Evaluación** de precisión sobre un conjunto de prueba.
- **Predicción** en tiempo real mediante una API.
- **Tests automáticos** con pytest.
- **CI/CD con GitHub Actions**: cada push a `main` ejecuta tests, entrena el modelo y sube el artefacto resultante.

El dataset Iris es un clásico en machine learning: contiene 150 muestras de 3 especies de flores (setosa, versicolor, virginica) con 4 características numéricas (longitud/ancho de sépalo y pétalo).

## 🚀 Objetivo

Demostrar un flujo completo de **Machine Learning + Data Engineering + DevOps** usando herramientas modernas:

- **PySpark** para procesamiento distribuido y ML.
- **FastAPI** para servir el modelo como API.
- **GitHub Actions** para integración y entrega continua (sin Docker).

## 📁 Estructura del proyecto
```text
├── .github/workflows/ci.yml # Pipeline CI/CD
├── app/
│ └── main.py # API FastAPI
├── data/
│ ├── iris.csv # Dataset completo (150 filas)
│ └── test_data.csv # Datos de prueba para validación
├── src/
│ ├── data_preprocessing.py # Carga y transformación de datos
│ ├── train_model.py # Entrenamiento del modelo
│ └── predict.py # Funciones de predicción
├── tests/
│ ├── test_preprocessing.py # Tests de preprocesamiento
│ ├── test_training.py # Tests de entrenamiento
│ └── test_prediction.py # Tests de predicción con datos de prueba
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

## 🔄 CI/CD con GitHub Actions

El pipeline definido en .github/workflows/ci.yml se ejecuta automáticamente:
  - En cada push a main o pull request: se instalan dependencias y se ejecutan los tests.
  - Solo en push a main (si los tests pasan): se entrena el modelo y se sube el artefacto iris-model (carpeta models/) a GitHub Actions.

Puedes descargar el artefacto desde la pestaña Actions → selecciona el run → Artifacts.

## 🧠 Tecnologías utilizadas
  - Apache Spark / PySpark
  - FastAPI
  - pytest
  - GitHub Actions
  - Python 3.10

## 📄 Licencia

Este proyecto es de uso educativo y libre.
