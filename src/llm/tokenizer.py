import tiktoken

def count_tokens(text, model="gpt-4.1-mini"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))
