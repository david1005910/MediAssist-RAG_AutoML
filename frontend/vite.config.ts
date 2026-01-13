import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3003,
    proxy: {
      '/api': {
        target: 'http://localhost:8003',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://localhost:8003',
        changeOrigin: true,
      },
      '/docs': {
        target: 'http://localhost:8003',
        changeOrigin: true,
      },
      '/redoc': {
        target: 'http://localhost:8003',
        changeOrigin: true,
      },
      '/openapi.json': {
        target: 'http://localhost:8003',
        changeOrigin: true,
      },
    },
  },
})
