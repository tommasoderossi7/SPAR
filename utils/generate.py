"""Utility for running CoT generation experiments via OpenRouter."""

from __future__ import annotations
import argparse
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import requests

from utils.utils import (
    append_to_json_array,
    generate_timestamp_id,
    load_json_array,
)

# Model used for inference; update as needed before running experiments.
model_name = "deepseek/deepseek-r1-0528:free"
# "deepseek/deepseek-r1:free","qwen/qwen3-235b-a22b:free",
# "deepseek/deepseek-r1-distill-llama-70b:free","deepseek/deepseek-r1-0528-qwen3-8b:free",
# "openai/gpt-oss-120b:free"

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
ENV_FILENAME = ".env"
TOP_LOGPROBS = 10
PROMPTS_FILENAME = "data/prompts/prompts.json"
GENERATIONS_DIRNAME = "data/generations"
GENERATIONS_REGISTRY = "generations.json"
COT_FILENAME = "cots.json"
LOGITS_FILENAME = "logits.json"
USAGE_FILENAME = "usage.json"

REASONING_TYPES = {"reasoning", "chain_of_thought", "thinking", "thought"}


def load_env_file(env_path: Path) -> None:
    """Load simple KEY=VALUE pairs from a .env file into os.environ when missing."""
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_api_key() -> str:
    env_path = Path(__file__).resolve().parents[1] / ENV_FILENAME
    load_env_file(env_path)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Ensure it is present in the environment or .env file."
        )
    return api_key


def build_headers(api_key: str) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    referer = os.environ.get("OPENROUTER_SITE_URL")
    title = os.environ.get("OPENROUTER_APP_NAME")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return headers


def normalise_content(message_content: Any) -> str:
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, Iterable):
        parts: List[str] = []
        for item in message_content:
            if isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if isinstance(value, str):
                    parts.append(value)
                elif isinstance(value, list):
                    parts.append(normalise_content(value))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return ""


def decode_token(
    token_value: Any, bytes_value: Optional[Iterable[int]] = None
) -> Optional[str]:
    if isinstance(token_value, str):
        return token_value
    if isinstance(token_value, (bytes, bytearray)):
        try:
            return token_value.decode("utf-8", "replace")
        except Exception:
            return None
    if bytes_value and isinstance(bytes_value, Iterable):
        try:
            return bytes(int(b) for b in bytes_value).decode("utf-8", "replace")
        except Exception:
            return None
    return None


def build_candidate_entry(
    token_value: Any, logprob_value: Any, bytes_value: Optional[Iterable[int]] = None
) -> Dict[str, Any]:
    token_text = decode_token(token_value, bytes_value)
    if token_text is None and token_value is not None:
        token_text = str(token_value)
    probability = (
        math.exp(logprob_value) if isinstance(logprob_value, (int, float)) else None
    )
    return {
        "token": token_text,
        "logprob": logprob_value,
        "probability": probability,
    }


def parse_top_logprobs_block(block: Any) -> List[Dict[str, Any]]:
    if not block:
        return []

    entries: List[Dict[str, Any]] = []
    if isinstance(block, dict):
        for token_value, logprob_value in block.items():
            entries.append(build_candidate_entry(token_value, logprob_value))
        return entries

    if isinstance(block, list):
        for item in block:
            if isinstance(item, dict):
                token_value = item.get("token") or item.get("text")
                logprob_value = item.get("logprob")
                bytes_value = item.get("bytes")
                entries.append(
                    build_candidate_entry(token_value, logprob_value, bytes_value)
                )
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                token_value, logprob_value = item
                entries.append(build_candidate_entry(token_value, logprob_value))
    return entries


def extract_candidates_from_content_item(item: Any) -> List[Dict[str, Any]]:
    if not isinstance(item, dict):
        return []

    logits_list: List[Dict[str, Any]] = []
    logits_list.extend(parse_top_logprobs_block(item.get("top_logprobs")))
    return logits_list


def convert_logprobs_to_logits(
    logprob_block: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not logprob_block:
        return []

    entries: List[Dict[str, Any]] = []

    content_blocks = logprob_block.get("content")
    if isinstance(content_blocks, list) and content_blocks:
        for index, block in enumerate(content_blocks):
            logits_list = extract_candidates_from_content_item(block)
            if not logits_list and isinstance(block, dict):
                token_value = block.get("token") or block.get("text")
                logits_list.append(
                    build_candidate_entry(
                        token_value, block.get("logprob"), block.get("bytes")
                    )
                )
            entries.append({"token_index": index, "logits": logits_list})
        return entries

    tokens = logprob_block.get("tokens")
    if isinstance(tokens, list) and tokens:
        token_logprobs = logprob_block.get("token_logprobs") or []
        top_logprobs = logprob_block.get("top_logprobs") or []
        for index, token_value in enumerate(tokens):
            logprob_value = (
                token_logprobs[index] if index < len(token_logprobs) else None
            )
            logits_list: List[Dict[str, Any]] = []
            if token_value is not None or isinstance(logprob_value, (int, float)):
                logits_list.append(build_candidate_entry(token_value, logprob_value))
            top_block = top_logprobs[index] if index < len(top_logprobs) else None
            logits_list.extend(parse_top_logprobs_block(top_block))
            entries.append({"token_index": index, "logits": logits_list})
        return entries

    return entries


def flatten_fragment_text(fragment: Any) -> str:
    if isinstance(fragment, str):
        return fragment
    if isinstance(fragment, dict):
        parts: List[str] = []
        text_value = fragment.get("text")
        if isinstance(text_value, str):
            parts.append(text_value)
        content_value = fragment.get("content")
        if isinstance(content_value, list):
            parts.append("".join(flatten_fragment_text(item) for item in content_value))
        elif isinstance(content_value, str):
            parts.append(content_value)
        return "".join(parts)
    if isinstance(fragment, list):
        return "".join(flatten_fragment_text(item) for item in fragment)
    return ""


def split_think_tags(text: str) -> Dict[str, str]:
    start_tag = "<think>"
    end_tag = "</think>"
    start_index = text.find(start_tag)
    end_index = text.find(end_tag)
    if start_index != -1 and end_index != -1 and end_index > start_index:
        cot = text[start_index + len(start_tag) : end_index].strip()
        after = (text[:start_index] + text[end_index + len(end_tag) :]).strip()
        return {"CoT_text": cot, "after_CoT_text": after}
    return {"CoT_text": "", "after_CoT_text": ""}


def extract_response_segments(choice: Dict[str, Any]) -> Dict[str, str]:
    message = choice.get("message", {})
    content = message.get("content")
    reasoning = message.get("reasoning")

    return {
        "CoT_text": reasoning if isinstance(reasoning, str) else "",
        "after_CoT_text": content if isinstance(content, str) else "",
    }


def call_openrouter(
    prompt: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    top_logprobs: int,
    *,
    max_retries: int = 5,
    base_backoff: float = 2.0,
) -> Dict[str, Any]:
    headers = build_headers(api_key)
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "logprobs": True,
        "top_logprobs": top_logprobs,
    }

    attempt = 0
    while True:
        response = requests.post(
            OPENROUTER_API_URL, headers=headers, json=payload, timeout=60
        )
        if response.status_code != 429:
            response.raise_for_status()
            return response.json()

        if attempt >= max_retries:
            response.raise_for_status()
            return response.json()

        retry_after = response.headers.get("retry-after")
        if retry_after is not None:
            try:
                delay = float(retry_after)
            except ValueError:
                delay = base_backoff * (2**attempt)
        else:
            delay = base_backoff * (2**attempt)

        attempt += 1
        time.sleep(delay)


def get_prompt_text(prompt_id: str) -> str:
    base_dir = Path(__file__).resolve().parents[1]
    prompts_path = base_dir / PROMPTS_FILENAME
    prompts = load_json_array(prompts_path)
    for prompt in prompts:
        if isinstance(prompt, dict) and prompt.get("prompt_id") == prompt_id:
            text = prompt.get("prompt_text")
            if isinstance(text, str):
                return text
            break
    raise RuntimeError(f"Prompt ID '{prompt_id}' not found in {prompts_path}.")


def record_generation(
    generation_id: str,
    prompt_id: str,
    response_segments: Dict[str, str],
    logits: List[Dict[str, Any]],
    response_json: Dict[str, Any],
) -> None:
    base_dir = Path(__file__).resolve().parents[1]
    generations_dir = base_dir / GENERATIONS_DIRNAME
    generations_dir.mkdir(parents=True, exist_ok=True)

    generations_path = generations_dir / GENERATIONS_REGISTRY
    generations_entry = {
        "generation_id": generation_id,
        "model_name": model_name,
        "prompt_id": prompt_id,
    }
    append_to_json_array(generations_path, generations_entry)

    cot_path = generations_dir / COT_FILENAME
    cot_entry = {
        "generation_id": generation_id,
        "CoT_text": response_segments.get("CoT_text", ""),
        "after_CoT_text": response_segments.get("after_CoT_text", ""),
    }
    append_to_json_array(cot_path, cot_entry)

    logits_path = generations_dir / LOGITS_FILENAME
    logits_entry = {
        "generation_id": generation_id,
        "logits": logits,
    }
    append_to_json_array(logits_path, logits_entry)

    usage_path = generations_dir / USAGE_FILENAME
    usage_entry = {
        "generation_id": generation_id,
        "prompt_tokens": response_json.get("usage", {}).get("prompt_tokens"),
        "completion_tokens": response_json.get("usage", {}).get("completion_tokens"),
        "total_tokens": response_json.get("usage", {}).get("total_tokens"),
        "api_key": "personal" if get_api_key().endswith("d9a98e3f1c2") else "spar",
    }
    append_to_json_array(usage_path, usage_entry)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CoT traces and token logits via OpenRouter."
    )
    parser.add_argument(
        "--prompt-id",
        required=True,
        help="Identifier of the prompt to send to the reasoning model.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature passed to the model.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50000,
        help="Maximum tokens the model may generate.",
    )
    parser.add_argument(
        "--top-logprobs",
        type=int,
        default=TOP_LOGPROBS,
        help="Number of alternate tokens to include per position when available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = get_api_key()
    prompt_text = get_prompt_text(args.prompt_id)
    generation_id = generate_timestamp_id()

    response_json = call_openrouter(
        prompt=prompt_text,
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_logprobs=args.top_logprobs,
    )

    if not response_json.get("choices"):
        raise RuntimeError("OpenRouter response did not contain any choices")

    choice = response_json["choices"][0]
    response_segments = extract_response_segments(choice)
    logits = convert_logprobs_to_logits(choice.get("logprobs"))
    # print("logits", logits)
    # print("choice", choice)

    record_generation(
        generation_id=generation_id,
        prompt_id=args.prompt_id,
        response_segments=response_segments,
        logits=logits,
        response_json=response_json,
    )
    print(f"Saved generation {generation_id}")


if __name__ == "__main__":
    main()
