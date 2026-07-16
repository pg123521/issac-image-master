import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  root: "web-static",
  publicDir: "../public",
  base: "/issac-image-master/",
  plugins: [react()],
  build: {
    outDir: "../dist-pages",
    emptyOutDir: true,
    target: "es2022",
  },
});
