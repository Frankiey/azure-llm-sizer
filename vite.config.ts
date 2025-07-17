import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  // Base path required for correct asset loading on GitHub Pages.
  base: '/azure-llm-sizer/',
  plugins: [react()],
})
