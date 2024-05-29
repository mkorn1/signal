import { FC } from "react"
import { EmotionThemeProvider } from "../components/Theme/EmotionThemeProvider"
import { GlobalCSS } from "../components/Theme/GlobalCSS"
import { ThemeContext } from "../hooks/useTheme"
import { LocalizationContext } from "../localize/useLocalization"
import { defaultTheme } from "../theme/Theme"
import { SignInPage } from "./SignInPage"

export const App: FC = () => {
  return (
    <ThemeContext.Provider value={defaultTheme}>
      <EmotionThemeProvider>
        <LocalizationContext.Provider value={{ language: null }}>
          <GlobalCSS />
          <SignInPage />
        </LocalizationContext.Provider>
      </EmotionThemeProvider>
    </ThemeContext.Provider>
  )
}
