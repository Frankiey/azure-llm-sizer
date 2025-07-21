import { useEffect, useMemo, useState } from 'react';
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

type SortOption = 'size_desc' | 'size_asc' | 'name';

const precisions: Precision[] = ['fp16', 'fp32', 'int8', 'int4'];
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

const MAX_MEM = 160; // for progress bar scaling

function App() {
  const [modelId, setModelId] = useState<string>((models as ModelInfo[])[0].model_id);
  const [precision, setPrecision] = useState<Precision>('fp16');
  // default to 128k context length
  const [ctxIndex, setCtxIndex] = useState<number>(7);
  const [result, setResult] = useState<ReturnType<typeof estimateWithSku> | null>(null);
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [sortOption, setSortOption] = useState<SortOption>('size_desc');
  const [search, setSearch] = useState('');
  const [dropdownOpen, setDropdownOpen] = useState(false);

  const ctx = ctxOptions[ctxIndex];

  // read configuration from query parameters on initial load
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const queryModel = params.get('model');
    if (queryModel) {
      const found = (models as ModelInfo[]).find((m) => {
        const slug = m.model_id.split('/').pop()?.toLowerCase() ?? m.model_id.toLowerCase();
        return slug === queryModel.toLowerCase() || m.model_id.toLowerCase() === queryModel.toLowerCase();
      });
      if (found) {
        setModelId(found.model_id);
        setSearch(found.model_id.split('/').pop() ?? found.model_id);
      }
    }
    const queryPrec = params.get('prec') as Precision | null;
    if (queryPrec && precisions.includes(queryPrec)) {
      setPrecision(queryPrec);
    }
    const queryCtx = params.get('ctx');
    if (queryCtx) {
      const match = queryCtx.toLowerCase().match(/^(\d+)(k)?$/);
      if (match) {
        const val = parseInt(match[1], 10) * (match[2] ? 1024 : 1);
        const idx = ctxOptions.indexOf(val);
        if (idx !== -1) setCtxIndex(idx);
      }
    }
  }, []);

  // update the query string whenever relevant state changes
  useEffect(() => {
    const params = new URLSearchParams();
    params.set('model', modelId.split('/').pop() ?? modelId);
    params.set('prec', precision);
    params.set('ctx', formatCtx(ctx));
    const url = `${window.location.pathname}?${params.toString()}`;
    window.history.replaceState({}, '', url);
  }, [modelId, precision, ctx]);

  const sortedModels = useMemo(() => {
    const arr = (models as ModelInfo[]).slice();
    switch (sortOption) {
      case 'size_asc':
        arr.sort((a, b) => a.params_b - b.params_b);
        break;
      case 'name':
        arr.sort((a, b) => a.model_id.localeCompare(b.model_id));
        break;
      default:
        arr.sort((a, b) => b.params_b - a.params_b);
        break;
    }
    return arr;
  }, [sortOption]);

  const filteredModels = useMemo(() => {
    const term = search.toLowerCase();
    return sortedModels.filter((m) => m.model_id.toLowerCase().includes(term));
  }, [search, sortedModels]);

  const highlightMatch = (name: string) => {
    if (!search) return name;
    const lower = name.toLowerCase();
    const term = search.toLowerCase();
    const index = lower.indexOf(term);
    if (index === -1) return name;
    return (
      <>
        {name.slice(0, index)}
        <span className="highlight">{name.slice(index, index + term.length)}</span>
        {name.slice(index + term.length)}
      </>
    );
  };

  const calc = () => {
    const model = (models as ModelInfo[]).find((m) => m.model_id === modelId);
    if (!model) return;
    const input: EstimateFullInput = {
      params_b: model.params_b,
      layers: model.layers,
      hidden: model.hidden,
      ctx,
      batch: 1,
      precision,
      skus: skus as unknown as AzureGpuSku[],
    };
    setResult(estimateWithSku(input));
  };

  useEffect(() => {
    calc();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleCalc = () => {
    setLoading(true);
    setTimeout(() => {
      calc();
      setLoading(false);
    }, 1000);
  };

  const formatCtx = (v: number) => (v >= 1024 ? `${v / 1024}k` : v.toString());


  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <div className="container">
      <div className="header">
        <h1>🚀 Azure LLM Sizer</h1>
        <p>Calculate the optimal Azure VM configuration for your Large Language Model workloads</p>
      </div>

      <div className="layout">
        <div className="main-card select-panel">
          <div className="section">
            <h2 className="section-title">🤖 Select Model</h2>
          <div className="model-sort">
            <label className="control-label" htmlFor="sort-select">Sort By</label>
            <select
              id="sort-select"
              className="control-input"
              value={sortOption}
              onChange={(e) => setSortOption(e.target.value as SortOption)}
            >
              <option value="size_desc">Size ↓</option>
              <option value="size_asc">Size ↑</option>
              <option value="name">Name</option>
            </select>
          </div>
          <div className="model-select">
            <label className="control-label" htmlFor="model-input">Model</label>
            <input
              id="model-input"
              className="model-input"
              type="text"
              placeholder="Type to search..."
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setDropdownOpen(true);
              }}
              onFocus={() => setDropdownOpen(true)}
            />
            {dropdownOpen && (
              <div className="model-dropdown">
                {filteredModels.map((m) => {
                  const name = m.model_id.split('/').pop() ?? m.model_id;
                  return (
                    <div
                      key={m.model_id}
                      className={`dropdown-item${m.model_id === modelId ? ' active' : ''}`}
                      onClick={() => {
                        setModelId(m.model_id);
                        setSearch(name);
                        setDropdownOpen(false);
                        calc();
                      }}
                    >
                      <span className="model-name">{highlightMatch(name)}</span>
                      <span className="model-size">{m.params_b}B parameters</span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          </div>

        <div className="section">
          <h2 className="section-title">⚙️ Configuration</h2>
          <div className="controls">
            <div className="control-group">
              <label className="control-label">Precision</label>
              <select
                className="control-input"
                value={precision}
                onChange={(e) => setPrecision(e.target.value as Precision)}
              >
                {precisions.map((p) => (
                  <option key={p} value={p}>
                    {p.toUpperCase()}
                  </option>
                ))}
              </select>
            </div>
            <div className="control-group">
              <label className="control-label">
                Context Length: <span id="context-value">{formatCtx(ctx)}</span>
              </label>
              <div className="slider-container">
                <input
                  type="range"
                  className="slider"
                  min={0}
                  max={ctxOptions.length - 1}
                  step={1}
                  value={ctxIndex}
                  onChange={(e) => setCtxIndex(Number(e.target.value))}
                  style={{
                    '--progress': `${(ctxIndex / (ctxOptions.length - 1)) * 100}%`,
                  } as React.CSSProperties}
                />
                <div className="slider-labels">
                  {ctxOptions.map((v) => (
                    <span key={v}>{formatCtx(v)}</span>
                  ))}
                </div>
              </div>
            </div>
          </div>
          <button className={`calculate-btn${loading ? ' loading' : ''}`} onClick={handleCalc}>
            {loading ? 'Calculating...' : 'Calculate Requirements'}
          </button>
        </div>
        </div>

        <div className="main-card results-panel">
        {result && (
          <div className={`results${!loading ? ' show' : ''}`} id="results">
            <h2 className="section-title">📊 Results</h2>
            <div className="results-grid">
              <div className="result-item">
                <div className="result-icon">💾</div>
                <div className="result-label">Model Weights</div>
                <div className="result-value" id="weights-value">
                  {result.weights_gb.toFixed(2)} GB
                </div>
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{ width: `${Math.min((result.weights_gb / MAX_MEM) * 100, 100)}%` }}
                  />
                </div>
              </div>
              <div className="result-item">
                <div className="result-icon">🔄</div>
                <div className="result-label">KV Cache</div>
                <div className="result-value" id="kv-value">
                  {result.kv_gb.toFixed(2)} GB
                </div>
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{ width: `${Math.min((result.kv_gb / MAX_MEM) * 100, 100)}%` }}
                  />
                </div>
              </div>
              <div className="result-item">
                <div className="result-icon">📈</div>
                <div className="result-label">Total Memory</div>
                <div className="result-value" id="total-memory">
                  {result.total_gb.toFixed(2)} GB
                </div>
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{ width: `${Math.min((result.total_gb / MAX_MEM) * 100, 100)}%` }}
                  />
                </div>
              </div>
              <div className="result-item">
                <div className="result-icon">🎮</div>
                <div className="result-label">GPUs Required</div>
                <div className="result-value" id="gpu-count">
                  {result.sku ? `${result.gpus} / ${result.sku.gpus_per_vm} per VM` : 'N/A'}
                </div>
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{ width: `${result.sku ? (result.gpus / result.sku.gpus_per_vm) * 100 : 0}%` }}
                  />
                </div>
              </div>
            </div>

            {result.sku ? (
              <>
                <div className="recommendation">
                  <div className="recommendation-title">💡 Recommended Configuration</div>
                  <div className="recommendation-text" id="recommendation-text">
                    {`${result.sku.sku} (${result.sku.gpu_model} ${result.sku.vram_gb} GB) - Memory per GPU: ${(result.total_gb / result.gpus).toFixed(2)} GB`}
                  </div>
                </div>
                <div className="command-section">
                  <div className="command-header">
                    <div className="command-title">🔧 Azure CLI Command</div>
                    <button
                      className="copy-btn"
                      onClick={() => handleCopy(`az vm create --name llm --size ${result.sku!.sku} --image UbuntuLTS`)}
                    >
                      {copied ? 'Copied!' : 'Copy'}
                    </button>
                  </div>
                  <div className="command-text" id="command-text">
                    {`az vm create --name llm --size ${result.sku.sku} --image UbuntuLTS`}
                  </div>
                </div>
                <div className="docs-link">
                  <a href={result.sku.docs_url} target="_blank" rel="noopener noreferrer">
                    View Azure Learn docs for this VM family
                  </a>
                </div>
              </>
            ) : (
              <div className="recommendation">
                <div className="recommendation-title">No suitable SKU found</div>
              </div>
            )}
          </div>
        )}
      </div>
      </div>
    </div>
  );
}

export default App;
