import { ToastProvider } from "dialog-hooks"
import { FC } from "react"
import { HelmetProvider } from "react-helmet-async"
import { EmotionThemeProvider } from "../../main/components/Theme/EmotionThemeProvider"
import { Toast } from "../components/Toast"
import { StoreContext } from "../hooks/useStores"
import RootStore from "../stores/RootStore"
import { GlobalCSS } from "./GlobalCSS"
import { RootView } from "./RootView"

export const App: FC = () => {
  return (
    <StoreContext.Provider value={new RootStore()}>
      <EmotionThemeProvider>
        <HelmetProvider>
          <ToastProvider component={Toast}>
            <GlobalCSS />
            <RootView />
          </ToastProvider>
        </HelmetProvider>
      </EmotionThemeProvider>
    </StoreContext.Provider>
  )
}
