import { useState } from 'react';
import { estimate } from './estimator';
import type { EstimateInput } from "./estimator";
import './App.css';

function App() {
  const [total, setTotal] = useState<number | null>(null);

  const handleClick = () => {
    const input: EstimateInput = {
      params_b: 1,
      layers: 12,
      hidden: 768,
      ctx: 1024,
      batch: 1,
      precision: 'fp16',
    };
    const { total_gb } = estimate(input);
    setTotal(total_gb);
  };

  return (
    <div className="p-4">
      <h1 className="text-xl font-bold">Azure LLM Sizer</h1>
      <button onClick={handleClick} className="mt-2 px-2 py-1 border">
        Run estimator
      </button>
      {total !== null && <p>Total GB: {total.toFixed(2)}</p>}
    </div>
  );
}

export default App;
