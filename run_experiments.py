import subprocess
import sys

# Configuraciones de experimentos a ejecutar
configs = [
    # Regresión Logística
    {"model_type": "logistic", "maxIter": 10, "regParam": 0.0},
    {"model_type": "logistic", "maxIter": 50, "regParam": 0.0},
    {"model_type": "logistic", "maxIter": 100, "regParam": 0.1},
    # Random Forest
    {"model_type": "random_forest", "numTrees": 20, "maxDepth": 5},
    {"model_type": "random_forest", "numTrees": 50, "maxDepth": 10},
    {"model_type": "random_forest", "numTrees": 100, "maxDepth": 15},
    # Árbol de decisión
    {"model_type": "decision_tree", "maxDepth": 3},
    {"model_type": "decision_tree", "maxDepth": 5},
    {"model_type": "decision_tree", "maxDepth": 10},
]

for cfg in configs:
    # Construir comando con los parámetros del diccionario
    cmd = ["python", "-m", "src.train_model"]
    for key, value in cfg.items():
        cmd.append(f"--{key}")
        cmd.append(str(value))

    print(f"\n=== Ejecutando: {' '.join(cmd)} ===")
    # Ejecutar y esperar a que termine
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Mostrar salida estándar
    if result.stdout:
        print(result.stdout)
    # Mostrar errores si los hay
    if result.stderr:
        print("Errores:")
        print(result.stderr)
    # Verificar código de retorno
    if result.returncode != 0:
        print(f"!!! El experimento falló con código {result.returncode}")
        # Opcional: detener el bucle si falla
        # break