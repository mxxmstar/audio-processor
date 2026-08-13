import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// Tauri 期望固定的端口与严格的 CSP；这里按 Tauri 2 的推荐配置。
export default defineConfig({
  plugins: [svelte()],
  clearScreen: false,
  server: {
    port: 23334,
    strictPort: true,
    watch: {
      ignored: ["**/src-tauri/**"],
    },
  },
  preview: {
    port: 23335,
    strictPort: true,
  },
  build: {
    target: "es2021",
    minify: "esbuild",
    sourcemap: false,
  },
});
