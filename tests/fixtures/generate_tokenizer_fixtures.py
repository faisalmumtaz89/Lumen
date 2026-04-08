#!/usr/bin/env python3
"""
Generate tokenizer ground truth fixtures for Lumen's native BPE tokenizer.

Uses HuggingFace AutoTokenizer as the oracle. Produces JSON fixture files
consumed by Rust tests to verify token-for-token correctness.

Usage:
    source ~/.venvs/mlx-bench/bin/activate
    python tests/fixtures/generate_tokenizer_fixtures.py

Output:
    tests/fixtures/tokenizer_*.json
"""

import json
import sys
from pathlib import Path

import transformers
from transformers import AutoTokenizer

FIXTURE_DIR = Path(__file__).parent

# Models to generate fixtures for.
MODELS = [
    {
        "key": "tinyllama",
        "hf_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "gguf_model_type": "llama",
        "gguf_pre": "default",
    },
    {
        "key": "llama-3.1-8b",
        "hf_id": "unsloth/Meta-Llama-3.1-8B-Instruct",
        "gguf_model_type": "gpt2",
        "gguf_pre": "llama-bpe",
    },
    {
        "key": "qwen2.5-3b",
        "hf_id": "Qwen/Qwen2.5-3B-Instruct",
        "gguf_model_type": "gpt2",
        "gguf_pre": "qwen2",
    },
    {
        "key": "qwen3.5-9b",
        "hf_id": "Qwen/Qwen3.5-9B",
        "gguf_model_type": "gpt2",
        "gguf_pre": "qwen35",
    },
]

# Test corpus — 33 inputs covering edge cases.
TEST_INPUTS = [
    # Basic (5)
    ("basic_hello", "Hello, world!"),
    ("basic_capital", "The capital of France is"),
    ("basic_math", "1 + 1 = 2"),
    ("basic_empty", ""),
    ("basic_space", " "),
    # Unicode (6)
    ("unicode_accents", "caf\u00e9 r\u00e9sum\u00e9"),
    ("unicode_cjk", "\u5317\u4eac\u5e02"),
    ("unicode_arabic", "\u0645\u0631\u062d\u0628\u0627"),
    ("unicode_emoji", "\U0001f389\U0001f525\U0001f4bb"),
    ("unicode_mixed", "Hello \u4e16\u754c"),
    ("unicode_zwj", "\U0001f468\u200d\U0001f469\u200d\U0001f467\u200d\U0001f466"),
    # Whitespace (5)
    ("ws_double_space", "Hello,  world!"),
    ("ws_leading", "   leading"),
    ("ws_trailing", "trailing   "),
    ("ws_newlines", "\n\nfoo\n"),
    ("ws_contractions", "it's a test"),
    # Merge edges (4)
    ("merge_repeated", "aaaaaa"),
    ("merge_bigram", "abababab"),
    ("merge_special_text", "<|endoftext|>"),
    ("merge_chatml_text", "<|im_start|>system"),
    # Code (4)
    ("code_python", "def foo(x):\n    return x * 2\n"),
    ("code_rust", 'fn main() {\n    println!("hello");\n}'),
    ("code_json", '{"key": [1, 2, 3], "nested": {"a": null}}'),
    ("code_html", '<div class="test">&amp;</div>'),
    # Boundary (4)
    ("boundary_single_char", "a"),
    ("boundary_paragraph", "The quick brown fox jumps over the lazy dog. " * 11),
    ("boundary_special_ws", "\u200b\u00a0"),
    ("boundary_url", "https://example.com/path?q=hello+world&x=1#anchor"),
    # Contractions (3)
    ("contract_multi", "don't won't can't"),
    ("contract_im", "I'm"),
    ("contract_theyre", "they're"),
    # Normalization (2)
    ("norm_nfc", "caf\u00e9"),
    ("norm_nfd", "cafe\u0301"),
]


def generate_fixture(model_info: dict) -> dict:
    key = model_info["key"]
    hf_id = model_info["hf_id"]
    print(f"  Loading tokenizer: {hf_id}...", end="", flush=True)
    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    print(f" vocab_size={tok.vocab_size}")

    test_cases = []
    for test_id, text in TEST_INPUTS:
        # Encode without special tokens (raw BPE output).
        ids = tok.encode(text, add_special_tokens=False)
        # Decode back.
        decoded = tok.decode(ids, skip_special_tokens=False)
        # Check round-trip.
        re_encoded = tok.encode(decoded, add_special_tokens=False)
        round_trips = (re_encoded == ids)

        test_cases.append({
            "id": test_id,
            "input": text,
            "token_ids": ids,
            "decoded": decoded,
            "round_trip": round_trips,
            "num_tokens": len(ids),
        })

    # Chat template test (single user turn).
    chat_ids = None
    chat_text = None
    try:
        msgs = [{"role": "user", "content": "Hello"}]
        chat_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        chat_ids = tok.encode(chat_text, add_special_tokens=False)
    except Exception as e:
        print(f"    Chat template failed: {e}")

    fixture = {
        "model_key": key,
        "hf_model_id": hf_id,
        "gguf_model_type": model_info["gguf_model_type"],
        "gguf_pre": model_info["gguf_pre"],
        "vocab_size": tok.vocab_size,
        "bos_token_id": tok.bos_token_id,
        "eos_token_id": tok.eos_token_id,
        "generator_versions": {
            "transformers": transformers.__version__,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        },
        "test_cases": test_cases,
        "chat_template_test": {
            "messages": [{"role": "user", "content": "Hello"}],
            "expected_text": chat_text,
            "expected_token_ids": chat_ids,
        } if chat_ids else None,
    }
    return fixture


def main():
    print("Generating tokenizer fixtures...")
    print(f"  transformers version: {transformers.__version__}")
    print()

    for model_info in MODELS:
        fixture = generate_fixture(model_info)
        out_path = FIXTURE_DIR / f"tokenizer_{model_info['key'].replace('.', '_').replace('-', '_')}.json"
        with open(out_path, "w") as f:
            json.dump(fixture, f, indent=2, ensure_ascii=False)
        n_cases = len(fixture["test_cases"])
        print(f"  Wrote {out_path.name} ({n_cases} test cases, vocab={fixture['vocab_size']})")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
