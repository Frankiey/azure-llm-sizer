# Azure LLM Sizer

**Azure LLM Sizer** is a small web application that estimates the GPU memory requirements for running large language models on Azure. It is built with [Vite](https://vitejs.dev/), React and TypeScript.

Try the hosted version at [frankiey.github.io/azure-llm-sizer](https://frankiey.github.io/azure-llm-sizer).

This is an independent project and is not affiliated with Microsoft.

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

This project requires **Node.js 20** or later. If the `dev` command fails with
`vite: not found` or similar errors, make sure you are using Node 20 and have
installed the dependencies with `npm install`.

The app will be available at `http://localhost:5173`.

### Linting

```bash
npm run lint
```

### Building

```bash
npm run build
```

Production files are emitted to the `dist/` directory. Because the app is hosted under a sub-path on GitHub Pages, the built files expect to be served from `/azure-llm-sizer/`.
You can preview the build locally with:

```bash
npm run preview
```

Then open `http://localhost:4173/azure-llm-sizer/` in your browser to view the preview.

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

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
