import { createRoot } from "react-dom/client"
import { Experiments } from "./components/Experiments/Experiments"
import { GlobalCSS } from "./components/Theme/GlobalCSS"
import { ThemeProvider } from "./theme/ThemeProvider"

const container = document.getElementById("root")
if (container) {
  const root = createRoot(container)
  root.render(
    <ThemeProvider>
      <GlobalCSS />
      <Experiments />
    </ThemeProvider>
  )
}


