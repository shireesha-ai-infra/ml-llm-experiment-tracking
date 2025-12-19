import mlflow
import time

from src.llm.client import load_llm, run_prompt
from src.llm.tokenizer import count_tokens
from src.utils.cost_estimator import estimate_cost

mlflow.set_experiment("llm experiments")

PROMPT_PATH = "experiments/llm/prompt_v1.txt"
MODEL_NAME = "gpt-4.1-mini"

prompt = open(PROMPT_PATH).read()

client, model = load_llm(MODEL_NAME)

start = time.time()
response = run_prompt(client, model, prompt)
latency = time.time() - start

output_text = response.choices[0].message.content

input_tokens = count_tokens(prompt,model)
output_tokens = count_tokens(output_text, model)
total_tokens = input_tokens + output_tokens

cost = estimate_cost(total_tokens, model)

with mlflow.start_run(run_name="prompt_v1"):
    mlflow.log_param("prompt_version", "v1")
    mlflow.log_param("llm_model", "gpt-4.1-mini")

    mlflow.log_metric("latency_sec", latency)
    mlflow.log_metric("total_tokens_used", total_tokens)
    mlflow.log_metric("cost_usd", cost)

    mlflow.log_text(prompt, "prompt.txt")
    mlflow.log_text(output_text, "response.txt")