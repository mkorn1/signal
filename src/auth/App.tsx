import { FC } from "react"
import { LocalizedContext } from "../common/localize/useLocalization"
import { defaultTheme } from "../common/theme/Theme"
import { EmotionThemeProvider } from "../main/components/Theme/EmotionThemeProvider"
import { GlobalCSS } from "../main/components/Theme/GlobalCSS"
import { ThemeContext } from "../main/hooks/useTheme"
import { SignInPage } from "./SignInPage"

export const App: FC = () => {
  return (
    <ThemeContext.Provider value={defaultTheme}>
      <EmotionThemeProvider>
        <LocalizedContext.Provider value={{ language: null }}>
          <GlobalCSS />
          <SignInPage />
        </LocalizedContext.Provider>
      </EmotionThemeProvider>
    </ThemeContext.Provider>
  )
}
