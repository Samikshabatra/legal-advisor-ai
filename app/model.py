# app/model.py
import os
import json
import re
import requests
import time

API_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL   = "meta-llama/Meta-Llama-3-8B-Instruct:novita"

_headers = None


def load_model():
    """
    No local model to load.
    Just verify the HF API token works.
    """
    global _headers

    hf_token = os.environ.get("HF_TOKEN", None)
    if not hf_token:
        raise ValueError(
            "\n❌ HF_TOKEN not set!\n"
            "Run this in your terminal first:\n"
            "  Windows: set HF_TOKEN=hf_your_token_here\n"
            "  Linux:   export HF_TOKEN=hf_your_token_here"
        )

    _headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type":  "application/json"
    }

    print("[Model] Testing HF API connection...")
    try:
        test_response = requests.post(
            API_URL,
            headers=_headers,
            json={
                "model":      MODEL,
                "messages":   [{"role": "user", "content": "Reply with: ok"}],
                "max_tokens": 5
            },
            timeout=30
        )

        if test_response.status_code == 200:
            print(f"[Model] ✅ Connected to HF API!")
            print(f"[Model] Model: {MODEL}")

        elif test_response.status_code == 401:
            raise ValueError(
                "❌ Invalid HF token — check your HF_TOKEN value"
            )

        elif test_response.status_code == 403:
            raise ValueError(
                "❌ Access denied to Llama-3.\n"
                "Request access at: huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct\n"
                "Then wait for approval email from Meta."
            )

        else:
            print(f"[Model] Warning: API returned {test_response.status_code}")
            print(f"[Model] Response: {test_response.text[:200]}")

    except requests.exceptions.Timeout:
        raise ValueError("❌ API connection timed out — check your internet")
    except requests.exceptions.ConnectionError:
        raise ValueError("❌ Cannot reach HF API — check your internet connection")


def run_inference(messages: list, max_new_tokens: int = 250) -> str:
    """
    Send messages to Llama-3 via HF API and get response.

    messages format:
    [
        {"role": "system", "content": "..."},
        {"role": "user",   "content": "..."}
    ]
    """
    global _headers

    payload = {
        "model":       MODEL,
        "messages":    messages,
        "max_tokens":  max_new_tokens,
        "temperature": 0.05,
        "top_p":       0.9,
        "stream":      False
    }

    # Try up to 3 times
    for attempt in range(3):
        try:
            response = requests.post(
                API_URL,
                headers=_headers,
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                return content.strip()

            elif response.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"[Model] Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue

            else:
                print(f"[Model] API error {response.status_code}: {response.text[:200]}")
                return '{"harmful": false, "articles": []}'

        except requests.exceptions.Timeout:
            print(f"[Model] Timeout on attempt {attempt+1}")
            if attempt < 2:
                time.sleep(3)
            continue

        except Exception as e:
            print(f"[Model] Request error: {e}")
            return '{"harmful": false, "articles": []}'

    return '{"harmful": false, "articles": []}'


def parse_llm_json(raw: str) -> dict:
    """
    Extract JSON from LLM response.
    Handles cases where model adds explanation around JSON.
    """
    raw = raw.strip()

    # Remove markdown code blocks if present
    raw = re.sub(r'```json\s*', '', raw)
    raw = re.sub(r'```\s*',     '', raw)
    raw = raw.strip()

    # Try 1: direct parse
    try:
        result = json.loads(raw)
        return result
    except:
        pass

    # Try 2: find JSON object anywhere in response
    match = re.search(r'\{.*?\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    # Try 3: manually extract fields using regex
    harmful_match  = re.search(
        r'"harmful"\s*:\s*(true|false)', raw, re.IGNORECASE
    )
    articles_match = re.search(
        r'"articles"\s*:\s*\[([^\]]*)\]', raw, re.DOTALL
    )

    if harmful_match:
        harmful  = harmful_match.group(1).lower() == "true"
        articles = []
        if articles_match:
            raw_list = articles_match.group(1)
            articles = re.findall(
                r'"(Section\s+[\d.]+[^"]*)"', raw_list
            )
        return {"harmful": harmful, "articles": articles}

    # Fallback — safe default
    print(f"[Model] WARNING: Could not parse JSON from: {raw[:300]}")
    return {"harmful": False, "articles": []}