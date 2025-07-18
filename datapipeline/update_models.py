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
    # Small to medium models
    {"model_id": "microsoft/phi-1", "params_b": 1.3, "layers": 24, "hidden": 2048, "moe_active_ratio": 0},
    {"model_id": "stabilityai/stablelm-2-zephyr-1_6b", "params_b": 1.6, "layers": 24, "hidden": 2048, "moe_active_ratio": 0},
    {"model_id": "microsoft/phi-2", "params_b": 2.7, "layers": 32, "hidden": 2560, "moe_active_ratio": 0},
    {"model_id": "microsoft/phi-3-mini-128k-instruct", "params_b": 3.8, "layers": 32, "hidden": 3072, "moe_active_ratio": 0},
    {"model_id": "stabilityai/stablelm-zephyr-3b", "params_b": 3.0, "layers": 32, "hidden": 2560, "moe_active_ratio": 0},
    {"model_id": "EleutherAI/gpt-j-6B", "params_b": 6, "layers": 28, "hidden": 4096, "moe_active_ratio": 0},
    {"model_id": "mosaicml/mpt-7b", "params_b": 7, "layers": 32, "hidden": 4096, "moe_active_ratio": 0},
    {"model_id": "microsoft/Phi-3-small-8k-instruct", "params_b": 7, "layers": 32, "hidden": 4096, "moe_active_ratio": 0},
    {"model_id": "meta-llama/Llama-2-7b-hf", "params_b": 7, "layers": 32, "hidden": 4096, "moe_active_ratio": 0},
    {"model_id": "tiiuae/falcon-7b", "params_b": 7, "layers": 32, "hidden": 4544, "moe_active_ratio": 0},
    {"model_id": "mistralai/Mistral-7B-v0.1", "params_b": 7.3, "layers": 32, "hidden": 4096, "moe_active_ratio": 0},
    {"model_id": "google/gemma-7b", "params_b": 7.0, "layers": 28, "hidden": 4096, "moe_active_ratio": 0},
    {"model_id": "mistralai/Mixtral-8x7B", "params_b": 56, "layers": 32, "hidden": 4096, "moe_active_ratio": 0.23},
    {"model_id": "meta-llama/Meta-Llama-3-8B", "params_b": 8.0, "layers": 32, "hidden": 4096, "moe_active_ratio": 0},
    {"model_id": "meta-llama/Llama-2-13b-hf", "params_b": 13, "layers": 40, "hidden": 5120, "moe_active_ratio": 0},
    {"model_id": "microsoft/phi-4", "params_b": 14, "layers": 40, "hidden": 5120, "moe_active_ratio": 0},
    {"model_id": "EleutherAI/gpt-neox-20b", "params_b": 20, "layers": 44, "hidden": 6144, "moe_active_ratio": 0},
    {"model_id": "mosaicml/mpt-30b", "params_b": 30, "layers": 48, "hidden": 7168, "moe_active_ratio": 0},
    {"model_id": "microsoft/Phi-3.5-MoE-instruct", "params_b": 61, "layers": 32, "hidden": 4096, "moe_active_ratio": 0.125},
    {"model_id": "meta-llama/Meta-Llama-3-70B", "params_b": 70, "layers": 80, "hidden": 8192, "moe_active_ratio": 0},
    {"model_id": "meta-llama/Llama-2-70b-hf", "params_b": 70, "layers": 80, "hidden": 8192, "moe_active_ratio": 0},
    {"model_id": "tiiuae/falcon-180B", "params_b": 180, "layers": 120, "hidden": 12288, "moe_active_ratio": 0},
    {"model_id": "THUDM/GLM-130B", "params_b": 130, "layers": 70, "hidden": 12288, "moe_active_ratio": 0},
    {"model_id": "bigscience/bloom", "params_b": 176, "layers": 70, "hidden": 14336, "moe_active_ratio": 0},
]

OUTPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "models.json"

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


def main() -> None:
    rows = []
    for entry in CATALOG:
        row = derive_fields(entry)
        rows.append(row.model_dump())
    OUTPUT_FILE.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
