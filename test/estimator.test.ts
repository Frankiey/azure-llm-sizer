import { test } from 'node:test';
import assert from 'node:assert';
import { estimateWithSku } from '../src/estimator.ts';

test('selects SKU with least VRAM per GPU', () => {
  const skus = [
    { sku: 'big', gpu_model: 'A100', gpus_per_vm: 1, vram_gb: 80, docs_url: '' },
    { sku: 'small', gpu_model: 'A100', gpus_per_vm: 1, vram_gb: 40, docs_url: '' },
  ];
  const result = estimateWithSku({
    params_b: 1,
    layers: 1,
    hidden: 1024,
    ctx: 256,
    batch: 1,
    precision: 'fp16',
    skus,
  });
  assert.equal(result.sku?.sku, 'small');
});
