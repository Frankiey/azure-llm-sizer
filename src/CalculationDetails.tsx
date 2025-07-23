interface Props {
  onClose: () => void;
}

function CalculationDetails({ onClose }: Props) {
  return (
    <div className="relative glass-card rounded-2xl shadow-xl p-6 mt-4 fade-in">
      <button
        type="button"
        aria-label="Close details"
        className="absolute top-2 right-2 text-gray-500 hover:text-gray-700"
        onClick={onClose}
      >
        ✖
      </button>
      <h3 className="text-xl font-bold text-gray-800 mb-4">Calculation details</h3>
      <p className="text-gray-700 mb-2">
        Memory requirements are derived from your selected model configuration.
        Model weights are computed as <code>parameters × bytes-per-parameter</code>
        based on the chosen precision.
      </p>
      <p className="text-gray-700 mb-2">
        The KV cache consumes
        <code>2 × layers × context × hidden × bytes × batch</code> with an
        additional 20% overhead for miscellaneous allocations.
      </p>
      <p className="text-gray-700">
        This tool focuses on inference workloads. It finds the smallest Azure GPU
        VM SKU that can fit the total memory and reports the number of GPUs
        required.
      </p>
    </div>
  );
}

export default CalculationDetails;
