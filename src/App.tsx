import { useState } from 'react';
import models from '../data/models.json';
import skus from '../data/azure-gpus.json';
import type { EstimateFullInput, Precision, AzureGpuSku } from './estimator';
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
const ctxOptions = [
  1024,
  2048,
  4096,
  8192,
  16384,
  32768,
  65536,
  131072,
  262144,
];

function App() {
  const [modelId, setModelId] = useState<string>(models[0].model_id);
  const [precision, setPrecision] = useState<Precision>('fp16');
  const [ctxIndex, setCtxIndex] = useState<number>(2);
  const ctx = ctxOptions[ctxIndex];
  const [result, setResult] = useState<ReturnType<typeof estimateWithSku> | null>(null);
  const [loading, setLoading] = useState(false);

  const handleCalc = () => {
    const model = (models as ModelInfo[]).find((m) => m.model_id === modelId);
    if (!model) return;
    setLoading(true);
    setTimeout(() => {
      const input: EstimateFullInput = {
        params_b: model.params_b,
        layers: model.layers,
        hidden: model.hidden,
        ctx,
        batch: 1,
        precision,
        skus: skus as AzureGpuSku[],
      };
      setResult(estimateWithSku(input));
      setLoading(false);
    }, 200);
  };

  return (
    <div className="container">
      <h1>Azure LLM Sizer</h1>
      <div className="models">
        {(models as ModelInfo[]).map((m) => {
          const name = m.model_id.split('/').pop();
          return (
            <button
              key={m.model_id}
              className={
                'model-tile' + (m.model_id === modelId ? ' active' : '')
              }
              onClick={() => setModelId(m.model_id)}
            >
              <strong>{name}</strong>
              <span>{m.params_b}B</span>
            </button>
          );
        })}
      </div>
      <div className="form">
        <label>
          Precision
          <select value={precision} onChange={(e) => setPrecision(e.target.value as Precision)}>
            {precisions.map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
        </label>
        <label>
          Context length: {ctx / 1024}k
          <input
            type="range"
            min={0}
            max={ctxOptions.length - 1}
            step="1"
            list="ctxTicks"
            value={ctxIndex}
            onChange={(e) => setCtxIndex(Number(e.target.value))}
          />
          <datalist id="ctxTicks">
            {ctxOptions.map((v, i) => (
              <option key={i} value={i} label={`${v / 1024}k`} />
            ))}
          </datalist>
        </label>
        <button onClick={handleCalc} disabled={loading}>Calculate</button>
        {loading && <div className="spinner" />}
      </div>
      {result && !loading && (
        <div className="result">
          <div className="metric">
            <span role="img" aria-label="weights">📦</span>
            <div className="progress"><span style={{ width: `${(result.weights_gb / result.total_gb) * 100}%` }} /></div>
            <span>{result.weights_gb.toFixed(2)} GB</span>
          </div>
          <div className="metric">
            <span role="img" aria-label="kv-cache">🔑</span>
            <div className="progress"><span style={{ width: `${(result.kv_gb / result.total_gb) * 100}%` }} /></div>
            <span>{result.kv_gb.toFixed(2)} GB</span>
          </div>
          <div className="metric">
            <span role="img" aria-label="total">💾</span>
            <div className="progress"><span style={{ width: '100%' }} /></div>
            <span>{result.total_gb.toFixed(2)} GB</span>
          </div>
          {result.sku ? (
            <>
              <p>GPUs required: {result.gpus} / {result.sku.gpus_per_vm} per VM</p>
              <p>
                Recommended SKU:
                <a href={`https://azure.microsoft.com/en-us/pricing/details/virtual-machines/${result.sku.sku.toLowerCase()}`}>{result.sku.sku}</a>
                {' '}({result.sku.gpu_model} {result.sku.vram_gb} GB)
              </p>
              <div className="metric">
                <span role="img" aria-label="per-gpu">🖥️</span>
                <div className="progress"><span style={{ width: `${((result.total_gb / result.gpus) / result.sku.vram_gb) * 100}%` }} /></div>
                <span>{(result.total_gb / result.gpus).toFixed(2)} / {result.sku.vram_gb} GB</span>
              </div>
              {(() => {
                const cli = `az vm create --name llm --size ${result.sku!.sku} --image UbuntuLTS`;
                return (
                  <pre className="cli">
                    <button
                      className="copy-btn"
                      onClick={() => navigator.clipboard.writeText(cli)}
                    >
                      copy
                    </button>
                    {cli}
                  </pre>
                );
              })()}
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
