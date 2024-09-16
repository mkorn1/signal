import { ThemeProvider as EmotionThemeProvider } from "@emotion/react"
import { observer } from "mobx-react-lite"
import { useStores } from "../hooks/useStores"
import { themes } from "./Theme"

export const ThemeProvider = observer(
  ({ children }: { children: React.ReactNode }) => {
    const { themeStore } = useStores()

    return (
      <EmotionThemeProvider theme={themes[themeStore.themeType]}>
        {children}
      </EmotionThemeProvider>
    )
  },
)
