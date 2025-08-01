import { useEffect, useMemo, useState, useRef } from 'react';
import CalculationDetails from './CalculationDetails';
import models from '../data/models.json';
import skus from '../data/azure-gpus.json';
import type { EstimateFullInput, Precision, AzureGpuSku } from './estimator';
import { estimateWithSku } from './estimator';

interface ModelInfo {
  model_id: string;
  params_b: number;
  layers: number;
  hidden: number;
  moe_active_ratio: number;
  ctx_len?: number;
}

type SortOption = 'size_desc' | 'size_asc' | 'name';

const precisions: Precision[] = ['fp16', 'bf16', 'fp32', 'int8', 'int4'];
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

const ctxIndexFor = (v: number): number => {
  const idx = ctxOptions.findIndex((c) => c >= v);
  return idx === -1 ? ctxOptions.length - 1 : idx;
};

const MAX_MEM = 160; // for progress bar scaling

function App() {
  // read configuration from query parameters before initializing state
  const query = useMemo(() => {
    const params = new URLSearchParams(window.location.search);

    const result = {
      modelId: 'meta-llama/Meta-Llama-3-70B',
      precision: 'fp16' as Precision,
      ctxIndex: 0,
      search: '',
    };

    const queryModel = params.get('model');
    if (queryModel) {
      const found = (models as ModelInfo[]).find((m) => {
        const slug = m.model_id.split('/').pop()?.toLowerCase() ?? m.model_id.toLowerCase();
        return slug === queryModel.toLowerCase() || m.model_id.toLowerCase() === queryModel.toLowerCase();
      });
      if (found) {
        result.modelId = found.model_id;
        result.search = found.model_id.split('/').pop() ?? found.model_id;
      }
    }

    const modelInfo = (models as ModelInfo[]).find((m) => m.model_id === result.modelId);
    if (modelInfo?.ctx_len) {
      result.ctxIndex = ctxIndexFor(modelInfo.ctx_len);
    }

    const queryPrec = params.get('prec') as Precision | null;
    if (queryPrec && precisions.includes(queryPrec)) {
      result.precision = queryPrec;
    }

    const queryCtx = params.get('ctx');
    if (queryCtx) {
      const match = queryCtx.toLowerCase().match(/^(\d+)(k)?$/);
      if (match) {
        const val = parseInt(match[1], 10) * (match[2] ? 1024 : 1);
        const idx = ctxOptions.indexOf(val);
        if (idx !== -1) {
          const maxIdx = ctxIndexFor(modelInfo?.ctx_len ?? ctxOptions[ctxOptions.length - 1]);
          result.ctxIndex = Math.min(idx, maxIdx);
        }
      }
    }

    if (!result.search) {
      result.search = result.modelId.split('/').pop() ?? result.modelId;
    }
    return result;
  }, []);

  const [modelId, setModelId] = useState<string>(query.modelId);
  const [precision, setPrecision] = useState<Precision>(query.precision);
  // context length defaults to the model maximum unless overridden by query string
  const [ctxIndex, setCtxIndex] = useState<number>(query.ctxIndex);
  const [result, setResult] = useState<ReturnType<typeof estimateWithSku> | null>(null);
  const [loading, setLoading] = useState(false);
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [sortOption, setSortOption] = useState<SortOption>('size_desc');
  const [search, setSearch] = useState(query.search);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [showDetails, setShowDetails] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const lastUpdated = useMemo(() => new Date().toLocaleDateString(), []);

  const selectedModel = useMemo(() => {
    return (models as ModelInfo[]).find((m) => m.model_id === modelId)!;
  }, [modelId]);

  const maxCtxIdx = useMemo(
    () => ctxIndexFor(selectedModel.ctx_len ?? ctxOptions[ctxOptions.length - 1]),
    [selectedModel]
  );

  const maxCtxPercent = useMemo(
    () => (maxCtxIdx / (ctxOptions.length - 1)) * 100,
    [maxCtxIdx]
  );

  const ctx = Math.min(
    ctxOptions[ctxIndex],
    selectedModel.ctx_len ?? ctxOptions[ctxIndex]
  );

  useEffect(() => {
    setCtxIndex(maxCtxIdx);
  }, [modelId, maxCtxIdx]);

  useEffect(() => {
    if (ctxIndex > maxCtxIdx) setCtxIndex(maxCtxIdx);
  }, [maxCtxIdx, ctxIndex]);

  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
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
        <span className="bg-blue-100 font-semibold">{name.slice(index, index + term.length)}</span>
        {name.slice(index + term.length)}
      </>
    );
  };

  const calc = () => {
    const model = selectedModel;
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
  }, [modelId]);

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
      setToastMessage('Copied to clipboard!');
      setTimeout(() => setToastMessage(null), 2000);
    });
  };

  return (
    <>
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      <header className="text-center mb-12 text-white">
        <div className="inline-block p-6 rounded-2xl glass-card mb-6 hover-lift transition-all duration-300">
          <h1 className="text-5xl font-bold mb-3 gradient-text">🚀 Azure LLM Sizer</h1>
          <p className="text-xl text-gray-700 max-w-2xl mx-auto">
            Calculate the optimal Azure VM configuration for your Large Language Model workloads with precision and ease
          </p>
          <p className="text-sm text-gray-600 mt-2 italic">
            Estimates memory requirements for inference only
          </p>
        </div>
      </header>

      <div className="grid lg:grid-cols-5 gap-8 items-start">
        <div className="lg:col-span-2 space-y-6">
          <div className="relative z-10 glass-card rounded-2xl shadow-xl p-6 hover-lift transition-all duration-300">
            <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-3">
              🤖 <span>Model Selection</span>
            </h2>
            <div className="flex justify-between items-center mb-4">
              <label className="text-sm font-medium text-gray-600">Sort models by:</label>
              <select
                id="sort-select"
                className="border-2 border-gray-200 rounded-lg px-3 py-2 focus:border-blue-500 focus:outline-none transition-colors"
                value={sortOption}
                onChange={(e) => setSortOption(e.target.value as SortOption)}
              >
                <option value="size_desc">Size ↓</option>
                <option value="size_asc">Size ↑</option>
                <option value="name">Name</option>
              </select>
            </div>
            <div className="relative" ref={dropdownRef}>
              <label className="block mb-2 text-sm font-semibold text-gray-700">Select Model</label>
              <div className="relative">
                <input
                  id="model-input"
                  className="w-full border-2 border-gray-200 rounded-lg px-4 py-3 focus:border-blue-500 focus:outline-none transition-colors text-lg"
                  placeholder="Search for models..."
                  type="text"
                  value={search}
                  onChange={(e) => {
                    setSearch(e.target.value);
                    setDropdownOpen(true);
                  }}
                  onFocus={() => setDropdownOpen(true)}
                />
                <div className="absolute right-3 top-3 text-gray-400">🔍</div>
              </div>
              {dropdownOpen && (
                <div className="absolute z-30 w-full bg-white border-2 border-gray-200 rounded-lg shadow-xl mt-2 max-h-[36rem] overflow-y-auto">
                  {filteredModels.map((m) => {
                    const name = m.model_id.split('/').pop() ?? m.model_id;
                    return (
                      <div
                        key={m.model_id}
                        className={`px-4 py-3 border-b last:border-b-0 cursor-pointer transition-colors ${m.model_id === modelId ? 'bg-blue-50 text-blue-700' : 'hover:bg-blue-50'}`}
                        onMouseDown={() => {
                          setModelId(m.model_id);
                          setSearch(name);
                          setDropdownOpen(false);
                        }}
                      >
                        <div className="font-semibold text-gray-800">{highlightMatch(name)}</div>
                        <div className="text-sm text-gray-500">{m.params_b}B parameters</div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
            <div className="mt-4 text-center">
              <a
                id="hf-link"
                href={`https://huggingface.co/${selectedModel.model_id}`}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-block bg-gradient-to-r from-blue-600 to-purple-600 text-white font-bold px-4 py-2 rounded-lg shadow-lg hover:from-blue-700 hover:to-purple-700 transform hover:scale-105 transition-all duration-200 cursor-pointer"
              >
                🤗 View on Hugging Face
              </a>
            </div>
          </div>

          <div className="glass-card rounded-2xl shadow-xl p-6 hover-lift transition-all duration-300">
            <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-3">
              ⚙️ <span>Configuration</span>
            </h2>
            <div className="space-y-6">
              <div>
                <label className="block mb-2 text-sm font-semibold text-gray-700">Precision</label>
                <select
                  id="precision-select"
                  className="w-full border-2 border-gray-200 rounded-lg px-4 py-3 focus:border-blue-500 focus:outline-none transition-colors"
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

              <div>
                <label className="block mb-2 text-sm font-semibold text-gray-700">
                  Context Length: <span id="context-value" className="text-blue-600 font-bold">{formatCtx(ctx)}</span>
                </label>
                <div className="relative">
                  {selectedModel.ctx_len && maxCtxIdx < ctxOptions.length - 1 && (
                    <div
                      className="absolute -top-6 text-blue-600 text-xs font-semibold pointer-events-none transform -translate-x-1/2"
                      style={{ left: `${maxCtxPercent}%` }}
                    >
                      <span className="leading-none">limit</span>
                      <span className="block leading-none">▼</span>
                    </div>
                  )}
                  <input
                    id="context-slider"
                    className="w-full h-3 bg-gray-200 rounded-lg appearance-none slider"
                    type="range"
                    min={0}
                    max={ctxOptions.length - 1}
                    step={1}
                    value={ctxIndex}
                    onChange={(e) =>
                      setCtxIndex(Math.min(Number(e.target.value), maxCtxIdx))
                    }
                  />
                </div>
                <div className="relative mt-2 text-xs text-gray-500 h-4">
                  {ctxOptions.map((v, i) => (
                    <span
                      key={v}
                      className="absolute -translate-x-1/2"
                      style={{ left: `${(i / (ctxOptions.length - 1)) * 100}%` }}
                    >
                      {formatCtx(v)}
                    </span>
                  ))}
                </div>
              </div>

              <button
                id="calculate-btn"
                className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white font-bold py-4 px-6 rounded-lg shadow-lg hover:from-blue-700 hover:to-purple-700 transform hover:scale-105 transition-all duration-200 cursor-pointer"
                onClick={handleCalc}
                disabled={loading}
              >
                {loading ? '⏳ Calculating...' : '✨ Calculate Requirements'}
              </button>
            </div>
          </div>
        </div>

        <div className="lg:col-span-3">
          {result && (
            <div id="results" className="glass-card rounded-2xl shadow-xl p-6 hover-lift transition-all duration-300 fade-in">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-3">
                  📊 <span>Resource Requirements</span>
                </h2>
                <div className="flex gap-2">
                  <button
                    id="details-btn"
                    className="text-sm text-blue-600 hover:underline"
                    onClick={() => setShowDetails((v) => !v)}
                  >
                    {showDetails ? 'Hide details' : 'How it works?'}
                  </button>
                </div>
              </div>
              {showDetails && (
                <CalculationDetails
                  onClose={() => setShowDetails(false)}
                  model={selectedModel}
                  ctx={ctx}
                  precision={precision}
                  result={result}
                />
              )}
              <div className="grid md:grid-cols-2 gap-4 mb-8">
                <div className="bg-gradient-to-br from-green-50 to-green-100 p-4 rounded-xl border-l-4 border-green-500">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-2xl">💾</span>
                    <span className="text-xs text-gray-600 font-semibold">MODEL WEIGHTS</span>
                  </div>
                  <div className="text-2xl font-bold text-green-700 mb-2">{result.weights_gb.toFixed(2)} GB</div>
                  <div className="w-full h-2 bg-gray-200 rounded-full">
                    <div className="h-full bg-green-500 rounded-full progress-bar" style={{ width: `${Math.min((result.weights_gb / MAX_MEM) * 100, 100)}%` }} />
                  </div>
                </div>

                <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-4 rounded-xl border-l-4 border-blue-500">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-2xl">🔄</span>
                    <span className="text-xs text-gray-600 font-semibold">KV CACHE</span>
                  </div>
                  <div className="text-2xl font-bold text-blue-700 mb-2">{result.kv_gb.toFixed(2)} GB</div>
                  <div className="w-full h-2 bg-gray-200 rounded-full">
                    <div className="h-full bg-blue-500 rounded-full progress-bar" style={{ width: `${Math.min((result.kv_gb / MAX_MEM) * 100, 100)}%` }} />
                  </div>
                </div>

                <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-4 rounded-xl border-l-4 border-purple-500">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-2xl">📈</span>
                    <span className="text-xs text-gray-600 font-semibold">TOTAL MEMORY</span>
                  </div>
                  <div className="text-2xl font-bold text-purple-700 mb-2">{result.total_gb.toFixed(2)} GB</div>
                  <div className="w-full h-2 bg-gray-200 rounded-full">
                    <div className="h-full bg-purple-500 rounded-full progress-bar" style={{ width: `${Math.min((result.total_gb / MAX_MEM) * 100, 100)}%` }} />
                  </div>
                </div>

                <div className="bg-gradient-to-br from-orange-50 to-orange-100 p-4 rounded-xl border-l-4 border-orange-500">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-2xl">🎮</span>
                    <span className="text-xs text-gray-600 font-semibold">GPUS REQUIRED</span>
                  </div>
                  <div className="text-2xl font-bold text-orange-700 mb-2">{result.sku ? `${result.gpus} of ${result.sku.gpus_per_vm} GPUs in this VM are required` : 'N/A'}</div>
                  <div className="w-full h-2 bg-gray-200 rounded-full">
                    <div className="h-full bg-orange-500 rounded-full progress-bar" style={{ width: `${result.sku ? Math.min((result.gpus / result.sku.gpus_per_vm) * 100, 100) : 0}%` }} />
                  </div>
                </div>
              </div>

              {result.sku ? (
                <>
                  <div className="bg-gradient-to-r from-green-500 to-green-600 text-white p-6 rounded-xl shadow-lg mb-6">
                    <div className="flex items-center gap-3 mb-3">
                      <span className="text-2xl">💡</span>
                      <h3 className="text-xl font-bold">Recommended Configuration</h3>
                    </div>
                    <div className="text-lg">
                      <span className="font-semibold">{result.sku.sku}</span> ({result.sku.gpu_model} {result.sku.vram_gb} GB)
                    </div>
                    <div className="text-sm opacity-90 mt-2">
                      Memory per GPU: {(result.total_gb / result.gpus).toFixed(2)} GB
                    </div>
                  </div>

                  <div className="bg-gray-900 text-white p-6 rounded-xl shadow-lg mb-6">
                    <div className="flex justify-between items-center mb-3">
                      <div className="flex items-center gap-3">
                        <span className="text-xl">🔧</span>
                        <span className="text-lg font-semibold text-cyan-400">Azure CLI Command</span>
                      </div>
                      <button
                        id="copy-btn-vm"
                        className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors flex items-center gap-2"
                        onClick={() =>
                          handleCopy(`az vm create --name llm-vm --size ${result.sku!.sku} --image Ubuntu2204 --location swedencentral`)
                        }
                      >
                        📋 Copy
                      </button>
                    </div>
                    <div className="bg-gray-800 p-3 rounded-lg break-all text-sm" id="cli-command">
                      {`az vm create --name llm-vm --size ${result.sku.sku} --image Ubuntu2204 --location swedencentral`}
                    </div>
                  </div>

                  <div className="bg-gray-900 text-white p-6 rounded-xl shadow-lg mb-6">
                    <div className="flex justify-between items-center mb-3">
                      <div className="flex items-center gap-3">
                        <span className="text-xl">🧪</span>
                        <span className="text-lg font-semibold text-cyan-400">Azure ML CLI Command</span>
                      </div>
                      <button
                        id="copy-btn-aml"
                        className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors flex items-center gap-2"
                        onClick={() =>
                          handleCopy(
                            `az ml compute create --name llm-vm --size ${result.sku!.sku} --type ComputeInstance --resource-group <your-resource-group> --workspace-name <your-workspace-name> --location swedencentral`
                          )
                        }
                      >
                        📋 Copy
                      </button>
                    </div>
                    <div className="bg-gray-800 p-3 rounded-lg break-all text-sm" id="aml-cli-command">
                      {`az ml compute create --name llm-vm --size ${result.sku.sku} --type ComputeInstance --resource-group <your-resource-group> --workspace-name <your-workspace-name> --location swedencentral`}
                    </div>
                  </div>

                  <div className="flex flex-col sm:flex-row gap-4 justify-between items-center text-sm">
                    <a
                      href={result.sku.docs_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-600 hover:text-blue-800 underline font-medium"
                    >
                      📚 View Azure Documentation
                    </a>
                    <button
                      className="text-blue-600 hover:text-blue-800 underline font-medium transition-colors"
                      onClick={() => handleCopy(window.location.href)}
                    >
                      🔗 Copy share link
                    </button>
                    <div className="text-gray-600">
                      Last updated: <span id="last-updated">{lastUpdated}</span>
                    </div>
                  </div>
                </>
              ) : (
                <div className="bg-gradient-to-r from-green-500 to-green-600 text-white p-6 rounded-xl shadow-lg mb-6">
                  <div className="font-semibold">No suitable SKU found</div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
    {toastMessage && (
      <div className="fixed bottom-4 left-1/2 transform -translate-x-1/2 bg-green-600 text-white px-4 py-2 rounded-lg shadow-lg fade-in">
        {toastMessage}
      </div>
    )}
    </>
  );
}

export default App;
