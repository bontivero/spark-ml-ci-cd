import subprocess

configs = [
    {"maxIter": 5, "regParam": 0.0},
    {"maxIter": 10, "regParam": 0.0},
    {"maxIter": 20, "regParam": 0.0},
    {"maxIter": 10, "regParam": 0.1},
    {"maxIter": 10, "regParam": 0.5},
]

for cfg in configs:
    cmd = f"python -m src.train_model --maxIter {cfg['maxIter']} --regParam {cfg['regParam']}"
    print(f"Ejecutando: {cmd}")
    subprocess.run(cmd, shell=True)