interface Props {
  onClose: () => void;
}

function CalculationDetails({ onClose }: Props) {
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
      <p className="text-gray-800 mb-2">
        Memory requirements are derived from your selected model configuration.
        Model weights are computed as <code>parameters × bytes-per-parameter</code>
        based on the chosen precision.
      </p>
      <p className="text-gray-800 mb-2">
        The KV cache consumes
        <code>2 × layers × context × hidden × bytes × batch</code> with an
        additional 20% overhead for miscellaneous allocations.
      </p>
      <p className="text-gray-800">
        This tool focuses on inference workloads. It finds the smallest Azure GPU
        VM SKU that can fit the total memory and reports the number of GPUs
        required.
      </p>
    </div>
  );
}

export default CalculationDetails;
