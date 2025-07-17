import json
from pathlib import Path

RAW_FILE = Path(__file__).with_name('vms.json')
INTERMEDIATE_FILE = Path(__file__).with_name('parsed_gpus.json')
OUTPUT_FILE = Path(__file__).resolve().parents[1] / 'data' / 'azure-gpus.json'

# map VM family strings to (GPU model, memory per GPU in GB)
FAMILY_TO_GPU = {
    'Standard NCASv3_T4 Family': ('T4', 16),
    'StandardNCADSA100v4Family': ('A100', 40),
    'StandardNCadsH100v5Family': ('H100', 80),
    'standard NDAMSv4_A100Family': ('A100', 80),
    'standardNDSH100v5Family': ('H100', 80),
    'StandardNVADSA10v5Family': ('A10', 24),
    'standardNVSv4Family': ('MI25', 16),
}

def main():
    raw = json.loads(RAW_FILE.read_text())
    parsed = []
    for item in raw:
        sku = item.get('name')
        family = item.get('family')
        gpu_model, vram = FAMILY_TO_GPU.get(family, ('Unknown', 0))
        gpus = 0
        for cap in item.get('capabilities', []):
            if cap.get('name') == 'GPUs':
                try:
                    gpus = int(cap['value'])
                except ValueError:
                    gpus = 0
                break
        parsed.append({'sku': sku, 'family': family, 'gpu_model': gpu_model, 'gpus_per_vm': gpus, 'vram_gb': vram})

    # write intermediate file with family info
    INTERMEDIATE_FILE.write_text(json.dumps(parsed, indent=2))

    # reduce to golden dataset used by application
    golden = [
        {
            'sku': p['sku'],
            'gpu_model': p['gpu_model'],
            'gpus_per_vm': p['gpus_per_vm'],
            'vram_gb': p['vram_gb'],
        }
        for p in parsed if p['gpu_model'] != 'Unknown'
    ]
    golden.sort(key=lambda x: x['sku'])
    OUTPUT_FILE.write_text(json.dumps(golden, indent=2))

if __name__ == '__main__':
    main()
