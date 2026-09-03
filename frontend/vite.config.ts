import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const root = fileURLToPath(new URL(".", import.meta.url));

export default defineConfig({
  plugins: [react()],
  // Assets are served by FastAPI under /static, and GET / serves the built
  // index.html, so the production bundle must reference /static/... paths.
  base: "/static/",
  build: {
    outDir: resolve(root, "../src/sam3_studio/static"),
    emptyOutDir: true,
    sourcemap: false,
    target: "es2022",
  },
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      "/health": "http://127.0.0.1:7860",
      "/config": "http://127.0.0.1:7860",
      "/segment": "http://127.0.0.1:7860",
      "/outputs": "http://127.0.0.1:7860",
    },
  },
});
