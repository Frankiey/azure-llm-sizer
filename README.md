# Azure LLM Sizer

**Azure LLM Sizer** is a small web application that estimates the GPU memory requirements for running large language models on Azure. It is built with [Vite](https://vitejs.dev/), React and TypeScript.

## Project structure

- `src/` – React components and the memory estimation logic.
- `data/` – JSON files describing available Azure GPU SKUs and model metadata.
- `index.html` – entry point used by Vite.
- `.github/workflows/` – GitHub Actions workflow that builds the application and deploys it to GitHub Pages.

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
