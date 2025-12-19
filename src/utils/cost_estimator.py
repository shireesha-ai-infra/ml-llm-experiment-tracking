PRICING = {"gpt-4.1-mini":0.000002}

def estimate_cost(tokens: int, model: str) -> float:
    assert isinstance(tokens, (int, float)), "tokens must be numeric"
    assert model in PRICING, f"Unknown model: {model}"

    return float(tokens) * PRICING[model]