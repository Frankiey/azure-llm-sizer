import type { Precision, EstimateOutput } from './estimator';

interface ModelDetails {
  model_id: string;
  params_b: number;
  layers: number;
  hidden: number;
  moe_active_ratio: number;
}

interface Props {
  onClose: () => void;
  model: ModelDetails;
  ctx: number;
  precision: Precision;
  result: EstimateOutput;
}

const BYTES: Record<Precision, number> = {
  fp32: 4,
  fp16: 2,
  bf16: 2,
  int8: 1,
  int4: 0.5,
};

function CalculationDetails({ onClose, model, ctx, precision, result }: Props) {
  const bytes = BYTES[precision];

  return (
    <div className="relative bg-gradient-to-br from-yellow-50 to-yellow-100 p-4 rounded-xl border-l-4 border-yellow-500 shadow-lg fade-in mb-4">
      <button
        type="button"
        aria-label="Close details"
        className="absolute top-2 right-2 text-gray-500 hover:text-gray-700"
        onClick={onClose}
      >
        ✖
      </button>
      <div className="flex items-center gap-2 mb-2">
        <span className="text-xl">ℹ️</span>
        <h3 className="text-lg font-bold text-yellow-800">Calculation details</h3>
      </div>
      <p className="text-gray-800 mb-4">
        Memory requirements are derived from your selected model configuration.
        Model weights use <code>parameters × bytes-per-parameter</code>, where
        each parameter uses {bytes} bytes in {precision.toUpperCase()}.
      </p>
      <div className="text-gray-800 text-sm space-y-1 mb-4">
        <div>Model: <span className="font-medium">{model.model_id}</span></div>
        <div>Parameters: {model.params_b}B</div>
        <div>Layers: {model.layers}</div>
        <div>Hidden size: {model.hidden}</div>
        <div>Context length: {ctx}</div>
        <div>Batch size: 1</div>
      </div>
      <ul className="text-gray-800 text-sm space-y-1 mb-4 list-disc list-inside">
        <li>
          Weights: {model.params_b}B × {bytes} bytes ={' '}
          {result.weights_gb.toFixed(2)} GB
        </li>
        <li>
          KV cache: 2 × {model.layers} × {ctx} × {model.hidden} × {bytes} × 1 /
          1e9 = {result.kv_gb.toFixed(2)} GB
        </li>
        <li>
          20% overhead → total {result.total_gb.toFixed(2)} GB
        </li>
      </ul>
      <p className="text-gray-800">
        This tool focuses on inference workloads. It selects the smallest Azure
        GPU VM SKU that can fit the total memory.
      </p>
    </div>
  );
}

export default CalculationDetails;
