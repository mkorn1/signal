import { ThemeProvider as EmotionThemeProvider } from "@emotion/react"
import { createContext, useContext, useState } from "react"
import { themes, ThemeType } from "./Theme"

const SetThemeContext = createContext<{
  themeType: ThemeType
  setThemeType: (themeType: ThemeType) => void
}>({
  themeType: "dark",
  setThemeType: () => {},
})

export const ThemeProvider = ({ children }: { children: React.ReactNode }) => {
  const [themeType, setThemeType] = useState<ThemeType>("dark")

  return (
    <SetThemeContext.Provider value={{ themeType, setThemeType }}>
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
