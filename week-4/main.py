import sys
import requests
import json
from open_api import MODELS, get_response


def main():
    prompt = (
        "In one sentence, name a limitation of treating a model's internal "
        "representations as a complete explanation of its behavior."
    )
    failed = False
    for model in MODELS:
        print(f"\n{model}")
        try:
            res = get_response(model, prompt) #json
            with open(f"response_{model.replace('/', '_')}.txt", "w") as f:
                json.dump(res, f, indent=2)
            try:
                content = res["choices"][0]["message"]["content"]
            except (IndexError, KeyError, TypeError) as error:
                raise ValueError(
                    f"Unexpected response format from model {model}"
                ) from error
            print(content)
           

        except requests.HTTPError as error:
            failed = True
            response = error.response
            if response is None:
                details = str(error)
            else:
                try:
                    payload = response.json()
                except ValueError:
                    payload = {}
                api_error = payload.get("error") if isinstance(payload, dict) else None
                metadata = api_error.get("metadata") if isinstance(api_error, dict) else None
                details = metadata.get("raw") if isinstance(metadata, dict) else None
                details = details or (
                    api_error.get("message") if isinstance(api_error, dict) else None
                )
                details = details or f"HTTP {response.status_code}: {response.reason}"
            print(f"OpenRouter request failed: {details}", file=sys.stderr)
        except ValueError as error:
            failed = True
            details = str(error)
            print(f"OpenRouter request failed: {details}", file=sys.stderr)

    if failed:
        raise SystemExit(1)

if __name__ == "__main__":
    main()