# 🚀 Project: ML & LLM Experiment Tracking System with MLflow

## Overview
This project demonstrates an experiment tracking system for:
- Classical Machine Learning
- Neural Networks
- Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)

The goal is to ensure **reproducibility, comparability, and governance** across all AI experiments.


## Why This Project Matters
In real-world AI systems, failures often come from:
- Unknown model versions
- Untracked prompts
- Unmeasured latency and cost
- Inability to reproduce results

This system treats **models, prompts, and RAG configs as versioned artifacts**.


## What Is Tracked

### Parameters
- ML hyperparameters
- Prompt versions
- RAG chunk sizes and retrievers

### Metrics
- Accuracy, loss
- Latency
- Tokens used
- Cost per run

### Artifacts
- Trained models
- Prompt files
- Generated responses


## Tech Stack
- MLflow (tracking + UI)
- Scikit-learn
- PyTorch
- Python

## How to Run

### Install dependencies
pip install -r requirements.txt

### Start MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

### Run experiments
- python experiments/ml/train_baseline.py
- python experiments/ml/train_improved.py
- python experiments/nn/train_nn.py
- python experiments/llm/run_llm_experiment.py
- python experiments/rag/run_rag_experiment.py


# 🔍 Tracking & Analyzing Experiments in MLflow UI

- After running all experiments, open the MLflow UI:

mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

- Navigate to: http://127.0.0.1:5000

This UI is the single source of truth for all ML, LLM, and RAG experiments in this project.


## 1️⃣ Experiment Selection (Top Left)

MLflow groups runs into experiments. In this project, we will see:

| Experiment Name | Purpose                          |
|-----------------|----------------------------------|
| ml-experiments  | Classical machine learning models |
| nn-experiments  | Neural network training           |
| llm-experiments | Prompt-based LLM runs             |
| rag-experiments | RAG configuration experiments    |


## 2️⃣ Runs Table (Core Analysis View)

Each row represents one immutable experiment run.

Columns to Focus On

| Column      | What It Means                         | Why It Matters                          |
|-------------|---------------------------------------|------------------------------------------|
| Run Name    | Logical identifier for an experiment  | Enables easy comparison across runs      |
| Start Time  | Timestamp when the run was executed   | Helps track recency and experiment order |
| Params      | Hyperparameters and configuration     | Ensures full reproducibility             |
| Metrics     | Quantitative performance results      | Allows objective quality comparison      |
| Artifacts   | Stored outputs (models, prompts, logs)| Enables auditing and safe rollback       |


## 3️⃣ Comparing ML Models (ML Experiments)

### How to Compare

	1.	Select multiple runs

	2.	Click “Compare”

### What to Analyze

| Metric / Field           | Interpretation                     |
|--------------------------|-------------------------------------|
| Accuracy                 | Indicates overall model quality     |
| Params (C, max_iter)     | Explains why model performance changed |


## 4️⃣ Neural Network Training Curves

### In nn-experiments:
- Click a run
- Open Metrics → loss

### What we Learn
- Convergence behavior
- Overfitting signals
- Training stability

This helps answer:

“Why did we choose this architecture?”


## 5️⃣ LLM Prompt Tracking (LLMOps)

In llm-experiments:

### Parameters to Inspect

| Param          | Meaning          |
|----------------|------------------|
| prompt_version | Prompt iteration |
| llm_model      | Model used       |

### Metrics to Inspect

| Metric        | Why It Matters        |
|---------------|------------------------|
| latency_sec  | User experience        |
| tokens_used  | Primary cost driver    |
| cost_usd     | Budget control         |

### Artifacts 

Open Artifacts:
- prompt.txt
- response.txt

👉 This allows:
- Prompt reproducibility
- Offline evaluation
- Auditing LLM outputs


## 6️⃣ RAG Configuration Tracking

In rag-experiments:

### Parameters

| Param            | Meaning                 |
|------------------|--------------------------|
| chunk_size       | Context granularity      |
| retriever        | Retrieval strategy       |
| embedding_model  | Representation choice    |

### Metrics

| Metric            | Interpretation          |
|-------------------|-------------------------|
| answer_relevance  | Output quality          |
| latency_sec       | System responsiveness   |


Smaller chunks improve relevance but increase latency.

