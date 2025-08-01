# MoE‑Aware VRAM Estimation — Design Notes & Implementation

**Status:** implemented in estimator (proposal + patch below)  
**Scope:** inference + training, single‑VM selection first; room to expand to multi‑node/per‑GPU modeling  

---

## TL;DR

- **MoE models** have **many total parameters** (all experts) but **activate only a fraction per token** (e.g., Mixtral 8×7B activates 2 of 8 experts per token; DeepSeek‑V3 activates ~37B out of 671B) — improving compute efficiency while keeping overall model capacity high.  
- For **single‑VM sizing** (no offload), you must be able to **load the full model footprint** (all experts). The **working set** per step is smaller (shared + active experts).  
- We add two inputs:  
  - `moe_active_ratio` — fraction of **expert** parameters active per token (e.g., 0.25 for top‑2 of 8).  
  - `moe_expert_fraction` — fraction of **total** params that are expert params (default 0.90 if `moe_active_ratio > 0`).  
- We compute **active** and **footprint** weights; for single‑VM, **footprint dominates** (see proof).  
- This keeps dense models unchanged and makes MoE estimates realistic, while leaving headroom for future features (offload, activation memory, per‑GPU sizing).

---

## Background: MoE active vs. total parameters

MoE layers contain **multiple experts** with a **router** that selects **top‑k experts per token**. Only the selected experts’ parameters are used per token; however **all experts must be available in memory across GPUs** so that the router can pick any of them during inference.

- **Mixtral 8×7B:** top‑2 of 8 experts per token ([Mistral blog](https://mistral.ai/news/mixtral-of-experts/)). Clarifications: only some layers are replicated; total params ≈ **45–47B**, not 56B ([HF blog](https://huggingface.co/blog/mixtral)). Practical summaries cite **~46.7B total**, **~12.9B active** per token ([example 1](https://patmcguinness.substack.com/p/mixtral-8x7b-and-mixture-of-experts), [example 2](https://ghost.oxen.ai/arxiv-dives-mixture-of-experts-moe-with-mixtral-8x7b/)).  
- **DeepSeek‑V3:** **671B total**, **37B active** per token (model card: [HF](https://huggingface.co/deepseek-ai/DeepSeek-V3); summaries: [IBM](https://www.ibm.com/think/topics/deepseek), [FT](https://www.ft.com/content/c82933fe-be28-463b-8336-d71a2ff5bbbf)).  
- **Why experts must be present:** Expert‑parallel systems shard experts across GPUs; **all experts need to be available in GPU memory** for routing/batching, unless explicitly offloaded ([HarMoEny, 2025](https://arxiv.org/html/2506.12417v2), [MoEShard, 2025](https://euromlsys.eu/pdf/euromlsys25-12.pdf), [DeepSpeed MoE inference tutorial](https://www.deepspeed.ai/tutorials/mixture-of-experts-inference/)).  
- Canonical MoE references: **Sparsely‑Gated MoE** ([Shazeer et al., 2017](https://arxiv.org/abs/1701.06538)); **Switch Transformer** (Google, top‑1 routing; see HF’s overview: [blog](https://huggingface.co/blog/moe)).

---

## Definitions

Let:
- `params_b` — **total** parameters (billions)  
- `moe_active_ratio` ∈ [0,1] — fraction of **expert** parameters that are **active** per token (e.g., Mixtral top‑2/8 ⇒ 0.25; DeepSeek‑V3 ~37/671 ≈ 0.055)  
- `moe_expert_fraction` ∈ [0,1] — fraction of **total** parameters that reside in **experts** (default 0.90 when `moe_active_ratio>0`)  
- `precision` ⇒ bytes/param (`fp32`=4, `fp16|bf16`=2, `int8`=1, `int4`=0.5)  
- `layers`, `hidden`, `ctx`, `batch` — used for KV cache estimate  
- `training` — toggles optimizer state estimate

We partition parameters:
- `p_total = params_b`  
- `p_experts = p_total * f_exp` with `f_exp = moe_expert_fraction`  
- `p_shared  = p_total - p_experts`

**Active** parameters per step:
```
p_active = p_shared + moe_active_ratio * p_experts
```
(If `moe_active_ratio <= 0`, treat as dense: `p_active = p_total` and `f_exp = 0`.)

---

## Memory model (current)

Weights (GB):
```
W_active    = bytes * p_active
W_footprint = bytes * p_total
```

KV cache (GB):
```
K = (2 * layers * ctx * hidden * bytes * batch) / 1e9
```

Optimizer state (GB): for **training**, tie to **total** params unless explicitly modeling optimizer sharding/offload:
```
opt = training ? 2.5 * W_footprint : 0
```

Safety factor: multiply totals by **1.2×** (fragmentation, buffers, kernels).

**Single‑VM, no offload:** total is governed by **footprint**:
```
total_footprint_gb = 1.2 * (W_footprint + K + opt)
total_gb = total_footprint_gb  // see proof below
```
We still report `W_active` (and `total_active_gb`) for transparency, but **VM selection** uses footprint.

### Why footprint ≥ active (proof sketch)
- `W_footprint` uses **all params**; `W_active` uses **shared + fraction of experts** ⇒ `W_footprint ≥ W_active`  
- `K` and `opt` are identical across branches (we tie `opt` to total params)  
⇒ `total_footprint_gb ≥ total_active_gb` (hence `max(active, footprint)` = `footprint`).

> This changes only if you add **offload/partitioning** (persistent on‑GPU footprint < total) or **activation memory** (large batches/contexts) to the model; see *Future work*.

---

## TypeScript patch (drop‑in)

```ts
export type Precision = 'fp32' | 'fp16' | 'bf16' | 'int8' | 'int4';

export interface EstimateInput {
  params_b: number;        // total params (billions)
  layers: number;
  hidden: number;
  ctx: number;             // map ctx_len -> ctx if needed
  batch: number;
  precision: Precision;
  training?: boolean;

  // --- MoE additions ---
  moe_active_ratio?: number;    // fraction of expert params active per token
  moe_expert_fraction?: number; // fraction of total params in experts
}

const BYTES: Record<Precision, number> = {
  fp32: 4, fp16: 2, bf16: 2, int8: 1, int4: 0.5,
};

export interface EstimateOutput {
  weights_gb: number;          // active weights for MoE; total for dense
  kv_gb: number;
  total_gb: number;            // VM sizing (single-VM, no-offload)
  gpus: number;
  sku?: AzureGpuSku;
  // optional visibility:
  weights_active_gb?: number;
  weights_footprint_gb?: number;
  total_active_gb?: number;
  total_footprint_gb?: number;
}

export interface AzureGpuSku {
  sku: string; gpu_model: string; gpus_per_vm: number; vram_gb: number; docs_url: string;
}

export interface EstimateFullInput extends EstimateInput {
  skus: AzureGpuSku[];
  vram_per_gpu?: number;
}

export function estimate(input: EstimateInput): Omit<EstimateOutput, 'gpus' | 'sku'> {
  const bytes = BYTES[input.precision];
  const isMoe = (input.moe_active_ratio ?? 0) > 0;
  const fExp = isMoe ? (input.moe_expert_fraction ?? 0.90) : 0.0;
  const rAct = isMoe ? (input.moe_active_ratio as number) : 0.0;

  const pTotal = input.params_b;            // billions
  const pExperts = pTotal * fExp;
  const pShared  = pTotal - pExperts;

  const pActive = isMoe ? (pShared + rAct * pExperts) : pTotal;

  const W_active    = (pActive * 1e9 * bytes) / 1e9;
  const W_footprint = (pTotal  * 1e9 * bytes) / 1e9;

  const K = (2 * input.layers * input.ctx * input.hidden * bytes * input.batch) / 1e9;
  const opt = input.training ? 2.5 * W_footprint : 0;

  const total_active_gb    = 1.2 * (W_active + K + opt);
  const total_footprint_gb = 1.2 * (W_footprint + K + opt);

  // Single-VM selection → footprint governs if no offload is modeled
  return {
    weights_gb: W_active,
    kv_gb: K,
    total_gb: total_footprint_gb,
    weights_active_gb: W_active,
    weights_footprint_gb: W_footprint,
    total_active_gb,
    total_footprint_gb,
  };
}

export function estimateWithSku(input: EstimateFullInput): EstimateOutput {
  const base = estimate(input);
  let best: AzureGpuSku | undefined;
  let needed = 0;
  for (const sku of [...input.skus].sort((a, b) => a.vram_gb - b.vram_gb)) {
    const gpus = Math.ceil(base.total_gb / sku.vram_gb);
    if (gpus <= sku.gpus_per_vm) { best = sku; needed = gpus; break; }
  }
  return { ...base, gpus: needed, sku: best };
}
```

---

## How to set `moe_active_ratio` and `moe_expert_fraction`

- **`moe_active_ratio`:** if routing is top‑k over `E` experts per MoE layer, a **first approximation** is `k/E`.  
  - Mixtral 8×7B: `k=2`, `E=8` ⇒ **0.25**.  
  - DeepSeek‑V3: **~37/671 ≈ 0.055** (from its model card).  
  - Some models use **top‑1** (Switch Transformer‑style) ⇒ ratio ≈ `1/E`.
  - Note: load balancing, expert sizes, and layer‑wise differences can make the effective ratio deviate from this simple rule; empirical measurement is best.
- **`moe_expert_fraction`:** if unknown, **0.90** is a sane default for LLM MoEs where experts dominate parameter count (FFN replication). For Mixtral, public breakdowns imply experts comprise the majority of parameters; embeddings/attention/router are shared.

---

## Example (illustrative numbers)

**DeepSeek‑R1‑style input** (noting the public V3 ratios as a proxy):  
`params_b = 685`, `moe_active_ratio = 0.05`, `moe_expert_fraction = 0.90`, `precision = fp16`

- `p_shared = 68.5B`, `p_experts = 616.5B`  
- `p_active = 68.5 + 0.05×616.5 ≈ 99.325B`  
- `W_active ≈ 99.3 × 2 bytes ≈ 198.7 GB`  
- `W_footprint ≈ 685 × 2 bytes ≈ 1,370 GB`  
⇒ Working set is ~199 GB, but aggregate footprint you must load across GPUs is ~**1.37 TB** (weights only; add KV/opt + 1.2× headroom).

---

## Future work (when `max(active, footprint)` matters)

Add knobs that change the on‑GPU memory model so **active** can exceed the **persistent on‑GPU footprint**:

1. **Offload / expert buffering:** offload non‑active experts to CPU/NVMe or other nodes; model an **on‑GPU persistent fraction** and `max(persistent, active)` ([Huang et al., NeurIPS’24](https://www.seas.upenn.edu/~leebcc/documents/huang24-neurips.pdf)).  
2. **Activation memory:** add an explicit activations term that scales with `batch × ctx × layers`; for large contexts, activations can dominate.  
3. **Per‑GPU sizing:** move from aggregate VRAM to **per‑GPU peak** under expert‑parallel sharding and KV placement (EP/TP/PP/DP interactions; [DeepSpeed](https://deepspeed.readthedocs.io/en/latest/inference-init.html), [Megatron‑Core MoE API](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/moe.html)).  
4. **Router/imbalance overheads:** account for load‑balancing losses and expert‑specific memory footprints.  
5. **Quantization‑aware bytes/param:** per‑layer precision, expert‑specific quantization, and optimizer precision for training.

---

## References

- Mistral, **Mixtral of Experts** announcement: top‑2‑of‑8 routing — https://mistral.ai/news/mixtral-of-experts/  
- Hugging Face, **Welcome Mixtral**: total params clarification — https://huggingface.co/blog/mixtral  
- Examples citing **46.7B total / 12.9B active** — https://patmcguinness.substack.com/p/mixtral-8x7b-and-mixture-of-experts, https://ghost.oxen.ai/arxiv-dives-mixture-of-experts-moe-with-mixtral-8x7b/  
- DeepSeek‑V3 model card (**671B total / 37B active**) — https://huggingface.co/deepseek-ai/DeepSeek-V3 (summaries: IBM https://www.ibm.com/think/topics/deepseek, FT https://www.ft.com/content/c82933fe-be28-463b-8336-d71a2ff5bbbf)  
- DeepSpeed **MoE inference** tutorial (expert partitioning) — https://www.deepspeed.ai/tutorials/mixture-of-experts-inference/  
- **HarMoEny** (2025): “all experts need to be available in GPU memory” — https://arxiv.org/html/2506.12417v2  
- **MoEShard** (EuroMLSys’25): memory footprint vs. active experts; expert parallelism — https://euromlsys.eu/pdf/euromlsys25-12.pdf  
- **Sparsely‑Gated MoE**, Shazeer et al. (2017) — https://arxiv.org/abs/1701.06538  
- HF **MoE overview** — https://huggingface.co/blog/moe
