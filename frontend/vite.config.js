import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import cesium from 'vite-plugin-cesium'
import { fileURLToPath, URL } from 'node:url'
// https://vitejs.dev/config/
export default defineConfig({
  plugins: [vue(), cesium()],
    resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
    server: {
    proxy: {
      '/process_image': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
      '/wms': {
        target: 'https://ows.environnement.brussels/air',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/wms/, ''),
      },
    },
  },
})
