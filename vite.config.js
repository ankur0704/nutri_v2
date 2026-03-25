import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // All /analyze_emotion, /recommend_diet, etc. → backend
      '/analyze_emotion': 'http://localhost:8000',
      '/recommend_diet': 'http://localhost:8000',
      '/log_entry': 'http://localhost:8000',
      '/get_history': 'http://localhost:8000',
      '/analytics': 'http://localhost:8000',
    }
  }
})
