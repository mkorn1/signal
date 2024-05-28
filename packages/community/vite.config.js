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
        "lodash",
        "midifile-ts",
        "mobx",
        "mobx-react-lite",
        "color",
        "firebase",
        /^@firebase\/.+/,
        "react",
        "react-dom",
        "react/jsx-runtime",
        "react-helmet-async",
        "react-share",
        /^@emotion\/.+/,
        /^@radix-ui\/.+/,
        /^@signal-app\/.+/,
        "use-l10n",
        "mdi-react",
      ],
    },
  },
  plugins: [react(), dts({ rollupTypes: true })],
})
