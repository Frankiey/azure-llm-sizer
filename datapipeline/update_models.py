#!/usr/bin/env python3
"""Refresh models.json from the Hugging Face Hub."""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError
from pydantic import BaseModel, Field
import sys


CATALOG = [
    # Deepseek R1

    {"model_id": "deepseek-ai/DeepSeek-R1", "params_b": 685.0, "layers": 61, "hidden": 7168,  "moe_active_ratio": 0.05} ,

    # DeepSeek R1 distills
    {"model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "params_b": 32, "layers": 64, "hidden": 5120, "moe_active_ratio": 0.0, "ctx_len": 32768},
    # {"model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", "params_b": 70, "layers": None, "hidden": None, "moe_active_ratio": 0.0, "ctx_len": None},
    # {"model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "params_b": 14, "layers": None, "hidden": None, "moe_active_ratio": 0.0, "ctx_len": None},
    # {"model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "params_b": 8, "layers": None, "hidden": None, "moe_active_ratio": 0.0, "ctx_len": None},
    # {"model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "params_b": 1.5, "layers": None, "hidden": None, "moe_active_ratio": 0.0, "ctx_len": None},

    # DeepSeek V2 line
    # {"model_id": "deepseek-ai/DeepSeek-V2-Chat", "params_b": None, "layers": None, "hidden": None, "moe_active_ratio": 0.0, "ctx_len": None},
    # {"model_id": "deepseek-ai/DeepSeek-V2-Chat-0628", "params_b": None, "layers": None, "hidden": None, "moe_active_ratio": 0.0, "ctx_len": None},
    # {"model_id": "deepseek-ai/DeepSeek-V2-Lite", "params_b": None, "layers": None, "hidden": None, "moe_active_ratio": 0.0, "ctx_len": None},

    # DeepSeek V1 LLM
    # {"model_id": "deepseek-ai/deepseek-llm-67b-chat", "params_b": 67, "layers": None, "hidden": None, "moe_active_ratio": 0.0, "ctx_len": 32768},

    # DeepSeek Coder V2
    # {"model_id": "deepseek-ai/DeepSeek-Coder-V2-Instruct", "params_b": None, "layers": None, "hidden": None, "moe_active_ratio": 0.0, "ctx_len": None},
    # {"model_id": "deepseek-ai/DeepSeek-Coder-V2-Instruct-0724", "params_b": None, "layers": None, "hidden": None, "moe_active_ratio": 0.0, "ctx_len": None},
    # {"model_id": "deepseek-ai/DeepSeek-Coder-V2-Lite-Base", "params_b": None, "layers": None, "hidden": None, "moe_active_ratio": 0.0, "ctx_len": None},
    # {"model_id": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", "params_b": None, "layers": None, "hidden": None, "moe_active_ratio": 0.0, "ctx_len": None},

    # Qwen / QwQ
    {"model_id": "Qwen/QwQ-32B-Preview", "params_b": 32, "layers": 64, "hidden": 5120, "moe_active_ratio": 0.0, "ctx_len": 131072},

    {"model_id": "Qwen/Qwen2.5-72B", "params_b": 72, "layers": 80, "hidden": 8192, "moe_active_ratio": 0.0, "ctx_len": 131072},
    {"model_id": "Qwen/Qwen2.5-72B-Instruct", "params_b": 72, "layers": 80, "hidden": 8192, "moe_active_ratio": 0.0, "ctx_len": 131072},
    {"model_id": "Qwen/Qwen2.5-32B-Instruct", "params_b": 32, "layers": 48, "hidden": 6656, "moe_active_ratio": 0.0, "ctx_len": 131072},
    {"model_id": "Qwen/Qwen2.5-Coder-32B", "params_b": 32, "layers": 48, "hidden": 6656, "moe_active_ratio": 0.0, "ctx_len": 131072},
    {"model_id": "Qwen/Qwen2.5-Coder-32B-Instruct", "params_b": 32, "layers": 48, "hidden": 6656, "moe_active_ratio": 0.0, "ctx_len": 131072},
    {"model_id": "Qwen/Qwen2.5-Coder-7B", "params_b": 7, "layers": 32, "hidden": 4096, "moe_active_ratio": 0.0, "ctx_len": 131072},
    {"model_id": "Qwen/Qwen2.5-Coder-7B-Instruct", "params_b": 7, "layers": 32, "hidden": 4096, "moe_active_ratio": 0.0, "ctx_len": 131072},

    {"model_id": "Qwen/Qwen1.5-110B", "params_b": 110, "layers": 80, "hidden": 8192, "moe_active_ratio": 0.0, "ctx_len": 32768},
    {"model_id": "Qwen/Qwen1.5-110B-Chat", "params_b": 110, "layers": 80, "hidden": 8192, "moe_active_ratio": 0.0, "ctx_len": 32768},

    {"model_id": "Qwen/Qwen2-72B", "params_b": 72, "layers": 80, "hidden": 8192, "moe_active_ratio": 0.0, "ctx_len": 32768},
    {"model_id": "Qwen/Qwen2-72B-Instruct", "params_b": 72, "layers": 80, "hidden": 8192, "moe_active_ratio": 0.0, "ctx_len": 32768},

    {"model_id": "Qwen/Qwen2.5-VL-72B-Instruct", "params_b": 72, "layers": 80, "hidden": 8192, "moe_active_ratio": 0.0, "ctx_len": 131072},
    {"model_id": "Qwen/Qwen2.5-Math-72B", "params_b": 72, "layers": 80, "hidden": 8192, "moe_active_ratio": 0.0, "ctx_len": 131072},
    {"model_id": "Qwen/Qwen2.5-Math-72B-Instruct", "params_b": 72, "layers": 80, "hidden": 8192, "moe_active_ratio": 0.0, "ctx_len": 131072},
    {"model_id": "Qwen/Qwen2.5-Math-PRM-72B", "params_b": 72, "layers": 80, "hidden": 8192, "moe_active_ratio": 0.0, "ctx_len": 131072},

    {"model_id": "Qwen/Qwen1.5-14B-Chat", "params_b": 14, "layers": 40, "hidden": 5120, "moe_active_ratio": 0.0, "ctx_len": 32768},

    # Meta - Llama 3.x, 2
    {"model_id": "meta-llama/Llama-3.1-405B", "params_b": 405, "layers": 126, "hidden": 16384, "moe_active_ratio": 0.0, "ctx_len": 131072},
    {"model_id": "meta-llama/Llama-3.1-405B-Instruct", "params_b": 405, "layers": 126, "hidden": 16384, "moe_active_ratio": 0.0, "ctx_len": 131072},
    {"model_id": "NousResearch/Hermes-3-Llama-3.1-405B", "params_b": 405, "layers": 126, "hidden": 16384, "moe_active_ratio": 0.0, "ctx_len": 131072},

    {"model_id": "meta-llama/Llama-3.3-70B-Instruct", "params_b": 70, "layers": 80, "hidden": 8192, "moe_active_ratio": 0.0, "ctx_len": 131072},

    {"model_id": "meta-llama/Meta-Llama-3-70B", "params_b": 70, "layers": 80, "hidden": 8192, "moe_active_ratio": 0.0, "ctx_len": 8192},
    {"model_id": "meta-llama/Meta-Llama-3-70B-Instruct", "params_b": 70, "layers": 80, "hidden": 8192, "moe_active_ratio": 0.0, "ctx_len": 8192},
    {"model_id": "meta-llama/Meta-Llama-3-8B", "params_b": 8, "layers": 32, "hidden": 8192, "moe_active_ratio": 0.0, "ctx_len": 8192},
    {"model_id": "meta-llama/Meta-Llama-3-8B-Instruct", "params_b": 8, "layers": 32, "hidden": 8192, "moe_active_ratio": 0.0, "ctx_len": 8192},

    {"model_id": "meta-llama/Llama-3.2-90B-Vision", "params_b": 90, "layers": 80, "hidden": 12288, "moe_active_ratio": 0.0, "ctx_len": 131072},
    {"model_id": "meta-llama/Llama-3.2-90B-Vision-Instruct", "params_b": 90, "layers": 80, "hidden": 12288, "moe_active_ratio": 0.0, "ctx_len": 131072},
    {"model_id": "meta-llama/Llama-3.2-11B-Vision", "params_b": 11, "layers": 36, "hidden": 6656, "moe_active_ratio": 0.0, "ctx_len": 131072},
    {"model_id": "meta-llama/Llama-3.2-11B-Vision-Instruct", "params_b": 11, "layers": 36, "hidden": 6656, "moe_active_ratio": 0.0, "ctx_len": 131072},
    {"model_id": "meta-llama/Llama-3.2-3B", "params_b": 3, "layers": 28, "hidden": 3072, "moe_active_ratio": 0.0, "ctx_len": 131072},
    {"model_id": "meta-llama/Llama-3.2-3B-Instruct", "params_b": 3, "layers": 28, "hidden": 3072, "moe_active_ratio": 0.0, "ctx_len": 131072},
    {"model_id": "meta-llama/Llama-3.2-1B", "params_b": 1, "layers": 16, "hidden": 2048, "moe_active_ratio": 0.0, "ctx_len": 131072},
    {"model_id": "meta-llama/Llama-3.2-1B-Instruct", "params_b": 1, "layers": 16, "hidden": 2048, "moe_active_ratio": 0.0, "ctx_len": 131072},

    {"model_id": "meta-llama/Llama-2-70b-chat-hf", "params_b": 70, "layers": 80, "hidden": 8192, "moe_active_ratio": 0.0, "ctx_len": 4096},
    {"model_id": "meta-llama/Llama-2-13b-chat-hf", "params_b": 13, "layers": 40, "hidden": 5120, "moe_active_ratio": 0.0, "ctx_len": 4096},
    {"model_id": "meta-llama/Llama-2-7b-chat-hf", "params_b": 7, "layers": 32, "hidden": 4096, "moe_active_ratio": 0.0, "ctx_len": 4096},

    # Mixtral / Mistral / Codestral / Pixtral / NeMo
    {"model_id": "mistralai/Mixtral-8x22B-v0.1", "params_b": 176, "layers": 56, "hidden": 6144, "moe_active_ratio": 0.25, "ctx_len": 32768},
    {"model_id": "mistralai/Mixtral-8x22B-Instruct-v0.1", "params_b": 176, "layers": 56, "hidden": 6144, "moe_active_ratio": 0.25, "ctx_len": 32768},
    {"model_id": "mistralai/Mixtral-8x7B-v0.1", "params_b": 56, "layers": 32, "hidden": 4096, "moe_active_ratio": 0.25, "ctx_len": 32768},
    {"model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1", "params_b": 56, "layers": 32, "hidden": 4096, "moe_active_ratio": 0.25, "ctx_len": 32768},

    {"model_id": "mistralai/Mistral-7B-v0.1", "params_b": 7, "layers": 32, "hidden": 4096, "moe_active_ratio": 0.0, "ctx_len": 8192},
    {"model_id": "mistralai/Mistral-7B-Instruct-v0.2", "params_b": 7, "layers": 32, "hidden": 4096, "moe_active_ratio": 0.0, "ctx_len": 8192},

    {"model_id": "mistralai/Mistral-Large-Instruct-2407", "params_b": 123, "layers": 80, "hidden": 12288, "moe_active_ratio": 0.0, "ctx_len": 128000},
    {"model_id": "mistralai/Mistral-Large-Instruct-2411", "params_b": 123, "layers": 80, "hidden": 12288, "moe_active_ratio": 0.0, "ctx_len": 128000},

    {"model_id": "mistralai/Codestral-22B-v0.1", "params_b": 22, "layers": 52, "hidden": 6144, "moe_active_ratio": 0.0, "ctx_len": 32768},
    {"model_id": "mistralai/Mamba-Codestral-7B-v0.1", "params_b": 7, "layers": 32, "hidden": 4096, "moe_active_ratio": 0.0, "ctx_len": 32768},

    {"model_id": "mistralai/Pixtral-12B-2409", "params_b": 12, "layers": 40, "hidden": 5120, "moe_active_ratio": 0.0, "ctx_len": 32768},
    {"model_id": "mistralai/Pixtral-12B-Base-2409", "params_b": 12, "layers": 40, "hidden": 5120, "moe_active_ratio": 0.0, "ctx_len": 32768},
    {"model_id": "mistralai/Pixtral-Large-Instruct-2411", "params_b": 34, "layers": 64, "hidden": 8192, "moe_active_ratio": 0.0, "ctx_len": 32768},

    {"model_id": "mistralai/Mistral-Nemo-Base-2407", "params_b": 12, "layers": 40, "hidden": 5120, "moe_active_ratio": 0.0, "ctx_len": 32768},
    {"model_id": "mistralai/Mistral-Nemo-Instruct-2407", "params_b": 12, "layers": 40, "hidden": 5120, "moe_active_ratio": 0.0, "ctx_len": 32768},
    {"model_id": "nvidia/Mistral-NeMo-12B-Instruct", "params_b": 12, "layers": 40, "hidden": 5120, "moe_active_ratio": 0.0, "ctx_len": 32768},

    # Nemotron
    # {"model_id": "nvidia/Nemotron-4-340B-Base", "params_b": 340, "layers": None, "hidden": None, "moe_active_ratio": 0.0, "ctx_len": None},
    # {"model_id": "nvidia/Nemotron-4-340B-Instruct", "params_b": 340, "layers": None, "hidden": None, "moe_active_ratio": 0.0, "ctx_len": None},
    # {"model_id": "nvidia/Nemotron-4-340B-Reward", "params_b": 340, "layers": None, "hidden": None, "moe_active_ratio": 0.0, "ctx_len": None},

    # Cohere (Command family)
    # {"model_id": "CohereLabs/c4ai-command-r-plus-08-2024", "params_b": None, "layers": None, "hidden": None, "moe_active_ratio": 0.0, "ctx_len": None},
    {"model_id": "CohereForAI/command-r", "params_b": 104, "layers": 64, "hidden": 12288, "moe_active_ratio": 0.0, "ctx_len": 131072},
    {"model_id": "CohereForAI/c4ai-command-r7b-12-2024", "params_b": 7, "layers": 32, "hidden": 4096, "moe_active_ratio": 0.0, "ctx_len": 32768},
    {"model_id": "CohereLabs/c4ai-command-a-03-2025", "params_b": 104, "layers": 64, "hidden": 12288, "moe_active_ratio": 0.0, "ctx_len": 131072},

    # Microsoft Phi
    {"model_id": "microsoft/phi-4", "params_b": 14, "layers": 48, "hidden": 6144, "moe_active_ratio": 0.0, "ctx_len": 131072},
    {"model_id": "microsoft/Phi-3-medium-128k-instruct", "params_b": 14, "layers": 40, "hidden": 6144, "moe_active_ratio": 0.0, "ctx_len": 128000},
    {"model_id": "microsoft/Phi-3-mini-128k-instruct", "params_b": 3.8, "layers": 32, "hidden": 3072, "moe_active_ratio": 0.0, "ctx_len": 128000},

    # Databricks DBRX
    {"model_id": "databricks/dbrx-base", "params_b": 132, "layers": 40, "hidden": 12288, "moe_active_ratio": 0.25, "ctx_len": 32768},
    {"model_id": "databricks/dbrx-instruct", "params_b": 132, "layers": 40, "hidden": 12288, "moe_active_ratio": 0.25, "ctx_len": 32768},

    # AI21 Jamba
    {"model_id": "ai21labs/AI21-Jamba-Large-1.7", "params_b": 52, "layers": 48, "hidden": 8192, "moe_active_ratio": 0.0, "ctx_len": 200000},
    {"model_id": "ai21labs/AI21-Jamba-Mini-1.7", "params_b": 12, "layers": 32, "hidden": 4096, "moe_active_ratio": 0.0, "ctx_len": 200000},
    {"model_id": "ai21labs/AI21-Jamba-Large-1.6", "params_b": 52, "layers": 48, "hidden": 8192, "moe_active_ratio": 0.0, "ctx_len": 200000},
    {"model_id": "ai21labs/AI21-Jamba-Mini-1.6", "params_b": 12, "layers": 32, "hidden": 4096, "moe_active_ratio": 0.0, "ctx_len": 200000},
    {"model_id": "ai21labs/AI21-Jamba-Large-1.5", "params_b": 52, "layers": 48, "hidden": 8192, "moe_active_ratio": 0.0, "ctx_len": 200000},
    {"model_id": "ai21labs/AI21-Jamba-Mini-1.5", "params_b": 12, "layers": 32, "hidden": 4096, "moe_active_ratio": 0.0, "ctx_len": 200000},

    # Nous Hermes / DeepHermes
    {"model_id": "NousResearch/Hermes-3-Llama-3.1-70B", "params_b": 70, "layers": 80, "hidden": 8192, "moe_active_ratio": 0.0, "ctx_len": 131072},
    {"model_id": "NousResearch/Hermes-3-Llama-3.1-8B", "params_b": 8, "layers": 32, "hidden": 8192, "moe_active_ratio": 0.0, "ctx_len": 131072},
    {"model_id": "NousResearch/Hermes-3-Llama-3.2-3B", "params_b": 3, "layers": 28, "hidden": 3072, "moe_active_ratio": 0.0, "ctx_len": 131072},

    {"model_id": "NousResearch/DeepHermes-3-Mistral-24B-Preview", "params_b": 24, "layers": 40, "hidden": 5120, "moe_active_ratio": 0.0, "ctx_len": 32768},
    {"model_id": "NousResearch/DeepHermes-3-Llama-3-8B-Preview", "params_b": 8, "layers": 32, "hidden": 8192, "moe_active_ratio": 0.0, "ctx_len": 8192},
    {"model_id": "NousResearch/DeepHermes-3-Llama-3-3B-Preview", "params_b": 3, "layers": 28, "hidden": 3072, "moe_active_ratio": 0.0, "ctx_len": 8192},

    # OpenChat
    {"model_id": "openchat/openchat_3.5", "params_b": 7, "layers": 32, "hidden": 4096, "moe_active_ratio": 0.0, "ctx_len": 8192},
    {"model_id": "openchat/openchat-3.5-0106", "params_b": 7, "layers": 32, "hidden": 4096, "moe_active_ratio": 0.0, "ctx_len": 8192},

    # Google Gemma 2
    {"model_id": "google/gemma-2-27b", "params_b": 27, "layers": 46, "hidden": 6144, "moe_active_ratio": 0.0, "ctx_len": 8192},
    {"model_id": "google/gemma-2-27b-it", "params_b": 27, "layers": 46, "hidden": 6144, "moe_active_ratio": 0.0, "ctx_len": 8192},
    {"model_id": "google/gemma-2-9b", "params_b": 9, "layers": 42, "hidden": 5120, "moe_active_ratio": 0.0, "ctx_len": 8192},
    {"model_id": "google/gemma-2-9b-it", "params_b": 9, "layers": 42, "hidden": 5120, "moe_active_ratio": 0.0, "ctx_len": 8192}
]


OUTPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "models.json"
STAGING_FILE = Path(__file__).resolve().parent / "staging_candidates.json"

class ModelRow(BaseModel):
    model_id: str
    params_b: float
    layers: int
    hidden: int
    moe_active_ratio: float = Field(ge=0, le=1)


def fetch_config(repo: str) -> dict | None:
    try:
        path = hf_hub_download(repo, "config.json")
    except GatedRepoError:
        print(f"Skipping {repo}: gated repo", file=sys.stderr)
        return None
    except HfHubHTTPError as e:
        print(f"Failed to download {repo} config: {e}", file=sys.stderr)
        return None
    with open(path, "r") as fh:
        return json.load(fh)


def derive_fields(entry: dict) -> ModelRow:
    repo = entry["model_id"]
    cfg = fetch_config(repo)
    if cfg is None:
        return ModelRow(**entry)
    total = cfg.get("num_parameters") or cfg.get("n_params")
    if total is None:
        try:
            info = HfApi().model_info(repo)
            total = info.safetensors.get("total_parameters") if info.safetensors else None
        except HfHubHTTPError:
            total = None
    params_b = round(float(total) / 1e9, 1) if total else entry.get("params_b", 0)
    layers = cfg.get("num_hidden_layers") or cfg.get("n_layer") or entry.get("layers")
    hidden = cfg.get("hidden_size") or cfg.get("n_embd") or cfg.get("d_model") or entry.get("hidden")
    moe_active = cfg.get("moe_active_expert_size")
    ratio = round(moe_active / total, 2) if moe_active and total else entry.get("moe_active_ratio", 0)
    return ModelRow(model_id=repo, params_b=params_b, layers=layers, hidden=hidden, moe_active_ratio=ratio)


def load_candidate_catalog() -> list[dict]:
    catalog_map = {entry["model_id"]: entry for entry in CATALOG}
    entries: list[dict] = []
    seen: set[str] = set()
    if STAGING_FILE.exists():
        try:
            candidates = json.loads(STAGING_FILE.read_text())
        except json.JSONDecodeError:
            candidates = []
        for candidate in candidates:
            model_id = candidate.get("model_id")
            if not model_id or model_id in seen:
                continue
            seen.add(model_id)
            entry = {
                "model_id": model_id,
                "params_b": 0.0,
                "layers": 0,
                "hidden": 0,
                "moe_active_ratio": 0.0,
            }
            entry.update(catalog_map.get(model_id, {}))
            entries.append(entry)
    for model_id, entry in catalog_map.items():
        if model_id in seen:
            continue
        entries.append(entry)
    return entries


def main() -> None:
    rows = []
    for entry in load_candidate_catalog():
        row = derive_fields(entry)
        rows.append(row.model_dump())
    OUTPUT_FILE.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
