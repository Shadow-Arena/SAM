import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const root = fileURLToPath(new URL(".", import.meta.url));
const apiTarget = process.env.VITE_API_TARGET ?? "http://127.0.0.1:8000";

const proxy = {
  "/segment": apiTarget,
  "/health": apiTarget,
  "/config": apiTarget,
  "/outputs": apiTarget,
  "/docs": apiTarget,
  "/openapi.json": apiTarget,
};

export default defineConfig({
  plugins: [react()],
  // The frontend is a standalone app: it owns its output (dist/) and is
  // served by its own process (Vite dev/preview, or Nginx in Docker).
  base: "/",
  build: {
    outDir: resolve(root, "dist"),
    emptyOutDir: true,
    sourcemap: false,
    target: "es2022",
  },
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy,
  },
  preview: {
    host: "0.0.0.0",
    port: 7860,
    proxy,
    // Accept sandbox/preview hostnames (dev/preview environments only —
    // production uses Nginx, which has no such check).
    allowedHosts: [".e2b.app", ".local", "localhost", "127.0.0.1"],
  },
});
