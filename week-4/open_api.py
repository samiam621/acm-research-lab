import requests
import json
from config import OPEN_ROUTER_KEY


API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS = (
    "nvidia/nemotron-3.5-lightning:free",
    "cohere/north-mini-code:free",
)


def get_response(model: str, prompt: str) -> str:
    if not OPEN_ROUTER_KEY:
        raise RuntimeError("Set OPEN_ROUTER_KEY in your .env file before running.")

    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {OPEN_ROUTER_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "reasoning": {"enabled": False, "exclude": True},
        },
        timeout=60,
    )
    response.raise_for_status()
    result = response.json()

    return result
