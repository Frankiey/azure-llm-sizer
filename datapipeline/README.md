# Data pipeline overview (model ingestion)

## Why this exists
Our UI depends on `data/models.json`, which feeds the model sizer. Historically we curated a
manual list (`CATALOG`) in `datapipeline/update_models.py`. That doesn't scale: we miss new
models and it creates a single‑person bottleneck. The ingestion layer introduces a staged
candidate catalog that can be refreshed automatically and reviewed before it affects the UI.

This README documents **what the ingestion layer does** and, more importantly, **why** it was
designed this way so new contributors can quickly understand the trade‑offs.

## High‑level flow
1. `ingest_candidates.py` collects candidates from multiple sources and normalizes them to a
   shared schema.
2. The normalized list is written to `datapipeline/staging_candidates.json` for review.
3. `update_models.py` loads the staged candidates and augments the manual `CATALOG` before
   deriving model sizes and writing `data/models.json`.

## Candidate schema
Each candidate record is normalized to:

```
{ model_id, provider, license, source, popularity_score, tags }
```

**Key decisions**
- We store **only identity + provenance here**. Sizing (params/layers/etc) still comes from
  `update_models.py` so we don't mix ingestion with model‑config derivation.
- `popularity_score` is an intentionally **loose score**. It uses downloads or ranking values
  when available and defaults to `0` otherwise. That gives us a consistent sorting signal
  without forcing a single "correct" metric.
- `source` is additive (e.g., `huggingface_hub+openrouter_index`) to preserve provenance
  without duplicating entries.

## Sources (and why)
We require three categories of sources to avoid bias from a single ecosystem:

### 1) Hub listings
`fetch_huggingface_hub()` pulls the most‑downloaded text‑generation models from the Hugging
Face Hub. This gives us broad coverage and reliable popularity signals.

**Why:** the hub is the largest open model registry. Download counts are a pragmatic proxy for
“interest,” even if they’re not perfect.

### 2) Community rankings
`fetch_community_rankings()` uses a small, curated list to highlight models the community
considers important even if their download counts are lower or they’re newer.

**Why:** purely popularity‑driven lists can miss new or specialized models. A curated list
adds intentional coverage for models we want to track.

### 3) Vendor/benchmark indexes
`fetch_openrouter_index()` pulls a public index of models exposed through OpenRouter and maps
entries to Hugging Face model IDs where possible.

**Why:** this captures **vendor‑hosted** or **benchmark‑listed** models that may not be
obvious from hub listings alone, and provides a cross‑provider view of what’s being offered.

## Why a staging file?
`staging_candidates.json` is an explicit "review point." We can refresh it frequently, but we
retain a stable, inspectable artifact before running the sizing pipeline. This keeps the UI
stable and allows maintainers to review new candidates if a source suddenly changes.

## Integration with update_models.py
`update_models.py` merges staged candidates with the manual `CATALOG`:
- Staged entries provide the **candidate IDs**.
- The manual `CATALOG` overrides entries where we already know specific sizes or need to
  pin known‑good models.

**Why:** the manual list is still valuable for hard‑earned sizing metadata. We treat it as
an overlay, not a replacement, so existing sizing data isn't lost.

## Operational notes
- Run `python datapipeline/ingest_candidates.py` to refresh the staged file.
- Run `python datapipeline/update_models.py` to regenerate `data/models.json`.
- The ingestion script uses network requests; if a source is unavailable, it safely returns
  an empty list rather than failing the whole pipeline.
