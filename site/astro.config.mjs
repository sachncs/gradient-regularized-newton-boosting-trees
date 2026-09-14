import { defineConfig } from "astro/config";
import tailwind from "@astrojs/tailwind";
import sitemap from "@astrojs/sitemap";

export default defineConfig({
  site: "https://sachncs.github.io",
  base: "/gradient-regularized-newton-boosting-trees",
  integrations: [tailwind(), sitemap()],
  build: {
    inlineStylesheets: "auto",
    assets: "_assets",
  },
  vite: {
    build: {
      cssMinify: true,
    },
  },
  compressHTML: true,
});
