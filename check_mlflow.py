import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

experiments = client.search_experiments()
print(f"Se encontraron {len(experiments)} experimento(s):")
for exp in experiments:
    print(f"- ID: {exp.experiment_id}, Nombre: {exp.name}")
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    print(f"  Runs: {len(runs)}")
    for run in runs:
        print(f"    Run ID: {run.info.run_id}, Estado: {run.info.status}")
        print(f"    Métricas: {run.data.metrics}")
        print(f"    Parámetros: {run.data.params}")
