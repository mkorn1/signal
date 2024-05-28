import react from "@vitejs/plugin-react"
import { resolve } from "path"
import { defineConfig } from "vite"
import dts from "vite-plugin-dts"

export default defineConfig({
  build: {
    lib: {
      entry: resolve(__dirname, "src/index.tsx"),
      name: "signal-player",
      fileName: "index",
      formats: ["es"],
    },
    rollupOptions: {
      external: [
        "midifile-ts",
        "mobx",
        "firebase",
        /^@firebase\/.+/,
        "react",
        "react-dom",
        "react/jsx-runtime",
      ],
    },
  },
  plugins: [react(), dts({ rollupTypes: true })],
})
