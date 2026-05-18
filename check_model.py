from mlflow.tracking import MlflowClient

client = MlflowClient()

models = client.search_registered_models()

for m in models:
    print("Model Name:", m.name)