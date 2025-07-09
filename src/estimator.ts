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
}

export function estimate(input: EstimateInput): EstimateOutput {
  const bytes = BYTES[input.precision];
  const W = (input.params_b * 1e9 * bytes) / 1e9;
  const K =
    (2 * input.layers * input.ctx * input.hidden * bytes * input.batch) /
    1e9;
  const opt = input.training ? 2.5 * W : 0;
  const total = 1.2 * (W + K + opt);
  return { weights_gb: W, kv_gb: K, total_gb: total };
}
