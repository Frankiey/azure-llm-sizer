#!/usr/bin/env python3
"""Collect candidate LLMs from multiple sources and stage them for update_models."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

STAGING_FILE = Path(__file__).resolve().parent / "staging_candidates.json"

COMMUNITY_RANKINGS = [
    {"model_id": "meta-llama/Llama-3.1-70B-Instruct", "score": 98.0},
    {"model_id": "Qwen/Qwen2.5-72B-Instruct", "score": 96.5},
    {"model_id": "mistralai/Mistral-Large-Instruct-2411", "score": 95.0},
    {"model_id": "google/gemma-2-27b-it", "score": 92.0},
    {"model_id": "microsoft/phi-4", "score": 90.0},
]


@dataclass
class Candidate:
    model_id: str
    provider: str
    license: str | None
    source: str
    popularity_score: float
    tags: list[str]


def _provider_from_model_id(model_id: str) -> str:
    return model_id.split("/", 1)[0] if "/" in model_id else model_id


def _fetch_json(url: str, headers: dict[str, str] | None = None) -> dict:
    req = Request(url, headers=headers or {})
    with urlopen(req, timeout=10) as response:
        return json.load(response)


def fetch_huggingface_hub(limit: int = 50) -> list[Candidate]:
    api = HfApi()
    candidates: list[Candidate] = []
    try:
        models = api.list_models(
            filter="text-generation",
            sort="downloads",
            direction=-1,
            limit=limit,
        )
    except HfHubHTTPError:
        models = []
    for model in models:
        model_id = model.modelId
        card_data = getattr(model, "cardData", None) or {}
        candidates.append(
            Candidate(
                model_id=model_id,
                provider=_provider_from_model_id(model_id),
                license=card_data.get("license") or None,
                source="huggingface_hub",
                popularity_score=float(model.downloads or 0),
                tags=list({*(model.tags or []), "hub-listing"}),
            )
        )
    return candidates


def fetch_community_rankings() -> list[Candidate]:
    candidates: list[Candidate] = []
    for entry in COMMUNITY_RANKINGS:
        model_id = entry["model_id"]
        candidates.append(
            Candidate(
                model_id=model_id,
                provider=_provider_from_model_id(model_id),
                license=None,
                source="community_rankings",
                popularity_score=float(entry["score"]),
                tags=["community-ranking"],
            )
        )
    return candidates


def fetch_openrouter_index(limit: int = 80) -> list[Candidate]:
    candidates: list[Candidate] = []
    try:
        data = _fetch_json(
            "https://openrouter.ai/api/v1/models",
            headers={"User-Agent": "azure-llm-sizer"},
        )
    except Exception:
        return candidates
    for entry in data.get("data", [])[:limit]:
        model_id = entry.get("hugging_face_id") or ""
        if not model_id:
            continue
        modalities = entry.get("architecture", {}).get("input_modalities") or []
        candidates.append(
            Candidate(
                model_id=model_id,
                provider=_provider_from_model_id(model_id),
                license=None,
                source="openrouter_index",
                popularity_score=0.0,
                tags=list({"vendor-index", *modalities}),
            )
        )
    return candidates


def merge_candidates(*sources: Iterable[Candidate]) -> list[Candidate]:
    merged: dict[str, Candidate] = {}
    for source in sources:
        for candidate in source:
            existing = merged.get(candidate.model_id)
            if existing is None:
                merged[candidate.model_id] = candidate
                continue
            existing.tags = sorted(set(existing.tags).union(candidate.tags))
            existing_source = set(existing.source.split("+"))
            new_source = set(candidate.source.split("+"))
            existing.source = "+".join(sorted(existing_source.union(new_source)))
            existing.popularity_score = max(existing.popularity_score, candidate.popularity_score)
            if existing.license is None:
                existing.license = candidate.license
    return sorted(
        merged.values(),
        key=lambda item: (-item.popularity_score, item.model_id),
    )


def stage_candidates() -> list[Candidate]:
    candidates = merge_candidates(
        fetch_huggingface_hub(),
        fetch_community_rankings(),
        fetch_openrouter_index(),
    )
    STAGING_FILE.write_text(json.dumps([asdict(c) for c in candidates], indent=2))
    return candidates


def main() -> None:
    candidates = stage_candidates()
    print(f"Wrote {len(candidates)} candidates to {STAGING_FILE}")


if __name__ == "__main__":
    main()
