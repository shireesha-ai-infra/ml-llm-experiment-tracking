import mlflow

mlflow.set_experiment("rag-experiments")

with mlflow.start_run(run_name="rag_config_1"):
    mlflow.log_param("chunk_size", 512)
    mlflow.log_param("retriever","faiss")
    mlflow.log_param("embedding_model", "all-MiniLM")

    mlflow.log_metric("answer_relevance", 0.74)
    mlflow.log_metric("latency_sec", 1.3)
    