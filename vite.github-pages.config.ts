import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  root: "web-static",
  publicDir: "../public",
  base: "/isaac-item-lens-web/",
  plugins: [react()],
  build: {
    outDir: "../dist-pages",
    emptyOutDir: true,
    target: "es2022",
  },
});
