import { ThemeProvider } from "@emotion/react"
import { ToastProvider } from "dialog-hooks"
import { FC } from "react"
import { HelmetProvider } from "react-helmet-async"
import { Toast } from "../components/Toast"
import { StoreContext } from "../hooks/useStores"
import RootStore from "../stores/RootStore"
import { defaultTheme } from "../theme/Theme"
import { GlobalCSS } from "./GlobalCSS"
import { RootView } from "./RootView"

export const App: FC = () => {
  return (
    <StoreContext.Provider value={new RootStore()}>
      <ThemeProvider theme={defaultTheme}>
        <HelmetProvider>
          <ToastProvider component={Toast}>
            <GlobalCSS />
            <RootView />
          </ToastProvider>
        </HelmetProvider>
      </ThemeProvider>
    </StoreContext.Provider>
  )
}
