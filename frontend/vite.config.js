import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import cesium from 'vite-plugin-cesium'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [vue(), cesium()],
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
