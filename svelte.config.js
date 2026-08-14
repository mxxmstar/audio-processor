import { vitePreprocess } from "@sveltejs/vite-plugin-svelte";

// Svelte 5 配置：显式开启 runes 模式（使用 $state / $props / $derived 等新语法）
export default {
  preprocess: vitePreprocess(),
  compilerOptions: {
    runes: true,
  },
};
