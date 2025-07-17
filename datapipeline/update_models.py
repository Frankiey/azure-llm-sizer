#!/usr/bin/env python3
"""Refresh models.json from the Hugging Face Hub."""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from pydantic import BaseModel, Field

CATALOG = [
    "stabilityai/stablelm-2-zephyr-1_6b",
    "microsoft/phi-3-mini-128k-instruct",
    "stabilityai/stablelm-zephyr-3b",
    "mistralai/Mistral-7B-v0.1",
    "google/gemma-7b",
    "mistralai/Mixtral-8x7B",
    "meta-llama/Meta-Llama-3-8B",
    "mosaicml/mpt-30b",
    "meta-llama/Meta-Llama-3-70B",
    "tiiuae/falcon-180B",
]

OUTPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "models.json"

class ModelRow(BaseModel):
    model_id: str
    params_b: float
    layers: int
    hidden: int
    moe_active_ratio: float = Field(ge=0, le=1)


def fetch_config(repo: str) -> dict:
    path = hf_hub_download(repo, "config.json")
    with open(path, "r") as fh:
        return json.load(fh)


def derive_fields(repo: str) -> ModelRow:
    cfg = fetch_config(repo)
    total = cfg.get("num_parameters") or cfg.get("n_params")
    if total is None:
        info = HfApi().model_info(repo)
        total = info.safetensors.get("total_parameters") if info.safetensors else None
    params_b = round(float(total) / 1e9, 1) if total else 0
    layers = cfg.get("num_hidden_layers") or cfg.get("n_layer")
    hidden = cfg.get("hidden_size") or cfg.get("n_embd") or cfg.get("d_model")
    moe_active = cfg.get("moe_active_expert_size")
    ratio = round(moe_active / total, 2) if moe_active and total else 0
    return ModelRow(model_id=repo, params_b=params_b, layers=layers, hidden=hidden, moe_active_ratio=ratio)


def main() -> None:
    rows = [derive_fields(repo).dict() for repo in CATALOG]
    OUTPUT_FILE.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
