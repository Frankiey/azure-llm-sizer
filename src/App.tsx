import { useState } from 'react';
import models from '../data/models.json';
import skus from '../data/azure-gpus.json';
import type { EstimateFullInput, Precision } from './estimator';
import { estimateWithSku } from './estimator';
import './App.css';

interface ModelInfo {
  model_id: string;
  params_b: number;
  layers: number;
  hidden: number;
  moe_active_ratio: number;
}

const precisions: Precision[] = ['fp32', 'fp16', 'bf16', 'int8', 'int4'];

function App() {
  const [modelId, setModelId] = useState<string>(models[0].model_id);
  const [precision, setPrecision] = useState<Precision>('fp16');
  const [ctx, setCtx] = useState<number>(4096);
  const [result, setResult] = useState<ReturnType<typeof estimateWithSku> | null>(null);

  const handleCalc = () => {
    const model = (models as ModelInfo[]).find((m) => m.model_id === modelId);
    if (!model) return;
    const input: EstimateFullInput = {
      params_b: model.params_b,
      layers: model.layers,
      hidden: model.hidden,
      ctx,
      batch: 1,
      precision,
      skus: skus as any,
    };
    setResult(estimateWithSku(input));
  };

  return (
    <div className="container">
      <h1>Azure LLM Sizer</h1>
      <div className="form">
        <label>
          Model
          <input
            list="models"
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
          />
          <datalist id="models">
            {(models as ModelInfo[]).map((m) => (
              <option key={m.model_id} value={m.model_id} />
            ))}
          </datalist>
        </label>
        <label>
          Precision
          <select value={precision} onChange={(e) => setPrecision(e.target.value as Precision)}>
            {precisions.map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
        </label>
        <label>
          Context length: {ctx}
          <input
            type="range"
            min="1"
            max="256000"
            step="1"
            value={ctx}
            onChange={(e) => setCtx(Number(e.target.value))}
          />
        </label>
        <button onClick={handleCalc}>Calculate</button>
      </div>
      {result && (
        <div className="result">
          <p>Weights: {result.weights_gb.toFixed(2)} GB</p>
          <p>KV cache: {result.kv_gb.toFixed(2)} GB</p>
          <p>Total memory: {result.total_gb.toFixed(2)} GB</p>
          {result.sku ? (
            <>
              <p>GPUs required: {result.gpus} / {result.sku.gpus_per_vm} per VM</p>
              <p>Recommended SKU: <a href={`https://azure.microsoft.com/en-us/pricing/details/virtual-machines/${result.sku.sku.toLowerCase()}`}>{result.sku.sku}</a></p>
              <pre className="cli">az vm create --name llm --size {result.sku.sku} --image UbuntuLTS</pre>
            </>
          ) : (
            <p>No suitable SKU found</p>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
