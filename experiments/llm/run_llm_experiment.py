import mlflow
import time

mlflow.set_experiment("llm experiments")

prompt_path = "experiments/llm/prompt_v1.txt"
prompt = open(prompt_path).read()

start = time.time()
response = "MLflow helps track experiments like Git tracks code."
latency = time.time() - start

tokens_used = 120
cost = tokens_used * 0.000002

with mlflow.start_run(run_name="prompt_v1"):
    mlflow.log_param("prompt_version", "v1")
    mlflow.log_param("llm_model", "gpt-4.1-mini")

    mlflow.log_metric("latency_sec", latency)
    mlflow.log_metric("tokens_used", tokens_used)
    mlflow.log_metric("cost_usd", cost)

    mlflow.log_text(prompt, "prompt.txt")
    mlflow.log_text(response, "response.txt")