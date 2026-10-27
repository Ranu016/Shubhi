"""Client for the local llama.cpp chat server."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


CHAT_API_URL = "http://127.0.0.1:8080/v1/chat/completions"


def generate_response(prompt: str, max_tokens: int = 128) -> str:
    """Send a prompt to the local model and return its answer."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    body = {
        "model": "local-model",
        "messages": [{"role": "user", "content": f"{prompt.rstrip()} /no_think"}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    request = Request(
        CHAT_API_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Local model server returned HTTP {error.code}: {details}") from error
    except URLError as error:
        raise RuntimeError(
            "Cannot reach the local model server. Start llama-server on port 8080 first."
        ) from error

    try:
        return result["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("The local model server returned an unexpected response.") from error


if __name__ == "__main__":
    print(generate_response("In one sentence, what is a tensor?"))
