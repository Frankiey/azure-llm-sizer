function CalculationDetails() {
  return (
    <div className="glass-card rounded-2xl shadow-xl p-6 mt-4 fade-in">
      <h3 className="text-xl font-bold text-gray-800 mb-4">Calculation details</h3>
      <p className="text-gray-700 mb-2">
        Memory requirements are derived from your selected model configuration.
        Model weights are computed as <code>parameters × bytes-per-parameter</code>
        based on the chosen precision.
      </p>
      <p className="text-gray-700 mb-2">
        The KV cache consumes
        <code>2 × layers × context × hidden × bytes × batch</code> and an
        additional 20% overhead is added for miscellaneous allocations. When
        training, a further <code>2.5×</code> the model weights is reserved for
        optimizer state.
      </p>
      <p className="text-gray-700">
        The app then finds the smallest Azure GPU VM SKU whose VRAM can hold the
        total memory and reports the number of GPUs required.
      </p>
    </div>
  );
}

export default CalculationDetails;
