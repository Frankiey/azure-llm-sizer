import { test } from 'node:test';
import assert from 'node:assert';
import { estimateWithSku } from '../src/estimator.ts';
import models from '../data/models.json';

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

test('GPT-NeoX-20B model is included in catalog', () => {
  const gptNeoX = models.find(model => model.model_id === 'EleutherAI/gpt-neox-20b');
  
  assert.ok(gptNeoX, 'GPT-NeoX-20B model should be in the catalog');
  assert.strictEqual(gptNeoX.params_b, 20, 'Should have 20 billion parameters');
  assert.strictEqual(gptNeoX.layers, 44, 'Should have 44 layers');
  assert.strictEqual(gptNeoX.hidden, 6144, 'Should have hidden size of 6144');
  assert.strictEqual(gptNeoX.moe_active_ratio, 0.0, 'Should not be a MoE model');
});

test('GPT-NeoX-20B works with estimator', () => {
  const gptNeoX = models.find(model => model.model_id === 'EleutherAI/gpt-neox-20b');
  assert.ok(gptNeoX, 'GPT-NeoX-20B model should be in the catalog');

  const result = estimateWithSku({
    params_b: gptNeoX.params_b,
    layers: gptNeoX.layers,
    hidden: gptNeoX.hidden,
    ctx: 2048,
    batch: 1,
    precision: 'fp16',
    skus: [
      { sku: 'A100-80', gpu_model: 'A100', gpus_per_vm: 8, vram_gb: 80, docs_url: '' },
      { sku: 'A100-40', gpu_model: 'A100', gpus_per_vm: 8, vram_gb: 40, docs_url: '' },
    ],
  });

  assert.ok(result.weights_gb > 0, 'Should calculate weight memory');
  assert.ok(result.kv_gb > 0, 'Should calculate KV cache memory');
  assert.ok(result.total_gb > 0, 'Should calculate total memory');
  assert.ok(result.gpus > 0, 'Should require at least one GPU');
  assert.ok(result.sku, 'Should select an appropriate SKU');
});
