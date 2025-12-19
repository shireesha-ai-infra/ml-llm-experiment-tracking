from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def load_llm(model_name : str):
    client= OpenAI()
    return client, model_name

def run_prompt(client, model, prompt):
    response = client.chat.completions.create(
        model = model,
        messages=[{"role":"user", "content": prompt}],
        temperature=0.2
    )
    return response