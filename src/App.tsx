import { useEffect, useMemo, useState } from 'react';
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
  // read configuration from query parameters before initializing state
  const query = useMemo(() => {
    const params = new URLSearchParams(window.location.search);
    const result = {
      // default to Meta-Llama-3-70B
      modelId: 'meta-llama/Meta-Llama-3-70B',
      precision: 'fp16' as Precision,
      ctxIndex: 7,
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
        if (idx !== -1) result.ctxIndex = idx;
      }
    }
    return result;
  }, []);

  const [modelId, setModelId] = useState<string>(query.modelId);
  const [precision, setPrecision] = useState<Precision>(query.precision);
  // default to 128k context length unless overridden by query string
  const [ctxIndex, setCtxIndex] = useState<number>(query.ctxIndex);
  const [result, setResult] = useState<ReturnType<typeof estimateWithSku> | null>(null);
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [sortOption, setSortOption] = useState<SortOption>('size_desc');
  const [search, setSearch] = useState(query.search);
  const [dropdownOpen, setDropdownOpen] = useState(false);

  const ctx = ctxOptions[ctxIndex];


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
    <div className="max-w-7xl mx-auto p-6">
      <header className="text-center mb-8 text-white">
        <h1 className="text-4xl font-bold mb-2">🚀 Azure LLM Sizer</h1>
        <p className="text-lg">Calculate the optimal Azure VM configuration for your Large Language Model workloads</p>
      </header>

      <div className="flex flex-col md:flex-row gap-6 items-start">
        <div className="bg-white rounded shadow p-6 flex-1 md:max-w-[40%]">
          <div className="mb-6">
            <h2 className="text-xl font-semibold text-blue-600 mb-4 flex items-center gap-2">🤖 Select Model</h2>
            <div className="flex justify-end mb-4">
              <label htmlFor="sort-select" className="mr-2 text-sm font-medium">Sort By</label>
              <select
                id="sort-select"
                className="border rounded px-2 py-1"
                value={sortOption}
                onChange={(e) => setSortOption(e.target.value as SortOption)}
              >
                <option value="size_desc">Size ↓</option>
                <option value="size_asc">Size ↑</option>
                <option value="name">Name</option>
              </select>
            </div>
            <div className="relative">
              <label htmlFor="model-input" className="block mb-1 text-sm font-medium">Model</label>
              <input
                id="model-input"
                className="w-full border rounded px-3 py-2"
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
                <div className="absolute z-10 w-full bg-white border rounded shadow max-h-72 overflow-y-auto">
                  {filteredModels.map((m) => {
                    const name = m.model_id.split('/').pop() ?? m.model_id;
                    return (
                      <div
                        key={m.model_id}
                        className={`px-4 py-3 border-b last:border-b-0 cursor-pointer ${m.model_id === modelId ? 'bg-blue-50 text-blue-700' : 'hover:bg-gray-100'}`}
                        onClick={() => {
                          setModelId(m.model_id);
                          setSearch(name);
                          setDropdownOpen(false);
                          calc();
                        }}
                      >
                        <span className="block font-semibold">{highlightMatch(name)}</span>
                        <span className="text-xs text-gray-500">{m.params_b}B parameters</span>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>

          <div className="mb-6">
            <h2 className="text-xl font-semibold text-blue-600 mb-4 flex items-center gap-2">⚙️ Configuration</h2>
            <div className="flex flex-wrap gap-4 items-end mb-4">
              <div className="flex flex-col flex-1 min-w-[200px]">
                <label className="mb-1 text-sm font-medium">Precision</label>
                <select
                  className="border rounded p-2"
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
              <div className="flex flex-col flex-1 min-w-[200px]">
                <label className="mb-1 text-sm font-medium">
                  Context Length: <span id="context-value">{formatCtx(ctx)}</span>
                </label>
                <input
                  type="range"
                  className="w-full"
                  min={0}
                  max={ctxOptions.length - 1}
                  step={1}
                  value={ctxIndex}
                  onChange={(e) => setCtxIndex(Number(e.target.value))}
                />
                <div className="flex justify-between text-xs mt-2">
                  {ctxOptions.map((v) => (
                    <span key={v}>{formatCtx(v)}</span>
                  ))}
                </div>
              </div>
            </div>
            <button
              className="bg-blue-600 text-white px-4 py-2 rounded shadow hover:bg-blue-700 disabled:opacity-50"
              onClick={handleCalc}
              disabled={loading}
            >
              {loading ? 'Calculating...' : 'Calculate Requirements'}
            </button>
          </div>
        </div>

        <div className="bg-white rounded shadow p-6 flex-1">
          {result && (
            <div
              className={`transition ${!loading ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-5'}`}
              id="results"
            >
              <h2 className="text-xl font-semibold text-blue-600 mb-4 flex items-center gap-2">📊 Results</h2>
              <div className="grid grid-cols-2 gap-3 mb-4">
                <div className="bg-white p-3 rounded shadow">
                  <div className="text-lg mb-1">💾</div>
                  <div className="text-xs text-gray-500 mb-1">Model Weights</div>
                  <div className="font-bold text-blue-600" id="weights-value">
                    {result.weights_gb.toFixed(2)} GB
                  </div>
                  <div className="w-full h-1 bg-gray-200 rounded mt-1">
                    <div
                      className="h-full bg-green-500"
                      style={{ width: `${Math.min((result.weights_gb / MAX_MEM) * 100, 100)}%` }}
                    />
                  </div>
                </div>
                <div className="bg-white p-3 rounded shadow">
                  <div className="text-lg mb-1">🔄</div>
                  <div className="text-xs text-gray-500 mb-1">KV Cache</div>
                  <div className="font-bold text-blue-600" id="kv-value">
                    {result.kv_gb.toFixed(2)} GB
                  </div>
                  <div className="w-full h-1 bg-gray-200 rounded mt-1">
                    <div
                      className="h-full bg-green-500"
                      style={{ width: `${Math.min((result.kv_gb / MAX_MEM) * 100, 100)}%` }}
                    />
                  </div>
                </div>
                <div className="bg-white p-3 rounded shadow">
                  <div className="text-lg mb-1">📈</div>
                  <div className="text-xs text-gray-500 mb-1">Total Memory</div>
                  <div className="font-bold text-blue-600" id="total-memory">
                    {result.total_gb.toFixed(2)} GB
                  </div>
                  <div className="w-full h-1 bg-gray-200 rounded mt-1">
                    <div
                      className="h-full bg-green-500"
                      style={{ width: `${Math.min((result.total_gb / MAX_MEM) * 100, 100)}%` }}
                    />
                  </div>
                </div>
                <div className="bg-white p-3 rounded shadow">
                  <div className="text-lg mb-1">🎮</div>
                  <div className="text-xs text-gray-500 mb-1">GPUs Required</div>
                  <div className="font-bold text-blue-600" id="gpu-count">
                    {result.sku ? `${result.gpus} / ${result.sku.gpus_per_vm} per VM` : 'N/A'}
                  </div>
                  <div className="w-full h-1 bg-gray-200 rounded mt-1">
                    <div
                      className="h-full bg-green-500"
                      style={{ width: `${result.sku ? (result.gpus / result.sku.gpus_per_vm) * 100 : 0}%` }}
                    />
                  </div>
                </div>
              </div>

              {result.sku ? (
                <>
                  <div className="bg-green-600 text-white p-4 rounded shadow mb-4">
                    <div className="font-semibold mb-1 flex items-center gap-2">💡 Recommended Configuration</div>
                    <div id="recommendation-text">
                      {`${result.sku.sku} (${result.sku.gpu_model} ${result.sku.vram_gb} GB) - Memory per GPU: ${(result.total_gb / result.gpus).toFixed(2)} GB`}
                    </div>
                  </div>
                  <div className="bg-neutral-800 text-white p-4 rounded shadow mb-4">
                    <div className="flex justify-between items-center mb-2">
                      <div className="text-cyan-400 font-semibold flex items-center gap-2">🔧 Azure CLI Command</div>
                      <button
                        className="bg-blue-600 text-white rounded px-2 py-1"
                        onClick={() =>
                          handleCopy(`az vm create --name llm --size ${result.sku!.sku} --image UbuntuLTS`)
                        }
                      >
                        {copied ? 'Copied!' : 'Copy'}
                      </button>
                    </div>
                    <div className="break-all text-sm" id="command-text">
                      {`az vm create --name llm --size ${result.sku.sku} --image UbuntuLTS`}
                    </div>
                  </div>
                  <div className="text-right">
                    <a
                      href={result.sku.docs_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-600 hover:underline"
                    >
                      View Azure Learn docs for this VM family
                    </a>
                  </div>
                </>
              ) : (
                <div className="bg-green-600 text-white p-4 rounded shadow mb-4">
                  <div className="font-semibold">No suitable SKU found</div>
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
