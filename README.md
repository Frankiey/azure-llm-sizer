# Azure LLM Sizer

**Azure LLM Sizer** is a small web application that estimates the GPU memory requirements for running large language models on Azure. It is built with [Vite](https://vitejs.dev/), React and TypeScript.

## Project structure

- `src/` – React components and the memory estimation logic.
- `data/` – JSON files describing available Azure GPU SKUs and model metadata.
- `index.html` – entry point used by Vite.
- `.github/workflows/` – GitHub Actions workflow that builds the application and deploys it to GitHub Pages.
- `datapipeline/` – scripts for fetching VM information and producing the GPU SKU dataset.

## Local development

Install dependencies and start the dev server:

```bash
npm install
npm run dev
```

The app will be available at `http://localhost:5173`.

### Linting

```bash
npm run lint
```

### Building

```bash
npm run build
```

Production files are emitted to the `dist/` directory. You can preview the build locally with:

```bash
npm run preview
```

Deployment to GitHub Pages occurs automatically when changes are pushed to the `main` branch.

### Generating the GPU dataset

The raw VM SKU information is stored in `datapipeline/vms.json`. Run the Python
script in that folder to produce the golden dataset consumed by the
application:

```bash
python3 datapipeline/generate_golden_dataset.py
```

This command writes the parsed information to `datapipeline/parsed_gpus.json`
and updates `data/azure-gpus.json`.

### Updating the model catalogue

Run the helper script to refresh `data/models.json` with the latest metadata from the Hugging Face Hub:

```bash
python3 datapipeline/update_models.py
```

The script enforces the JSON schema with `pydantic` and fetches each model's configuration directly from the hub. Models that require accepting a license on Hugging Face are skipped during the update and keep their existing values.
