export type Precision = 'fp32' | 'fp16' | 'bf16' | 'int8' | 'int4';

export interface EstimateInput {
  params_b: number;
  layers: number;
  hidden: number;
  ctx: number;
  batch: number;
  precision: Precision;
  training?: boolean;
}

const BYTES: Record<Precision, number> = {
  fp32: 4,
  fp16: 2,
  bf16: 2,
  int8: 1,
  int4: 0.5,
};

export interface EstimateOutput {
  weights_gb: number;
  kv_gb: number;
  total_gb: number;
  gpus: number;
  sku?: AzureGpuSku;
}

export interface AzureGpuSku {
  sku: string;
  gpu_model: string;
  gpus_per_vm: number;
  vram_gb: number;
  docs_url: string;
}

export interface EstimateFullInput extends EstimateInput {
  skus: AzureGpuSku[];
  vram_per_gpu?: number;
}

export function estimate(input: EstimateInput): Omit<EstimateOutput, 'gpus' | 'sku'> {
  const bytes = BYTES[input.precision];
  const W = (input.params_b * 1e9 * bytes) / 1e9;
  const K =
    (2 * input.layers * input.ctx * input.hidden * bytes * input.batch) /
    1e9;
  const opt = input.training ? 2.5 * W : 0;
  const total = 1.2 * (W + K + opt);
  return { weights_gb: W, kv_gb: K, total_gb: total };
}

export function estimateWithSku(input: EstimateFullInput): EstimateOutput {
  const base = estimate(input);
  let best: AzureGpuSku | undefined;
  let needed = 0;
  for (const sku of [...input.skus].sort((a, b) => a.vram_gb - b.vram_gb)) {
    const gpus = Math.ceil(base.total_gb / sku.vram_gb);
    if (gpus <= sku.gpus_per_vm) {
      best = sku;
      needed = gpus;
      break;
    }
  }
  return { ...base, gpus: needed, sku: best };
}

