import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ command }) => ({
  // Use the GitHub Pages base path for production builds only.
  base: command === 'build' ? '/azure-llm-sizer/' : '/',
  plugins: [react()],
}))
