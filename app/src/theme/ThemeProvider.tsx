import { ThemeProvider as EmotionThemeProvider } from "@emotion/react"
import { createContext, useContext, useState } from "react"
import { themes, ThemeType } from "./Theme"

const SetThemeContext = createContext<{
  setThemeType: (themeType: ThemeType) => void
}>({
  setThemeType: () => {},
})

export const ThemeProvider = ({ children }: { children: React.ReactNode }) => {
  const [themeType, setThemeType] = useState<ThemeType>("dark")

  return (
    <SetThemeContext.Provider value={{ setThemeType }}>
      <EmotionThemeProvider theme={themes[themeType]}>
        {children}
      </EmotionThemeProvider>
    </SetThemeContext.Provider>
  )
}

export const useSetTheme = () => {
  const context = useContext(SetThemeContext)
  if (!context) {
    throw new Error("useSetTheme must be used within a ThemeProvider")
  }
  return context
}
