import React from "react"
import { HelmetProvider } from "react-helmet-async"
import { defaultTheme } from "../../../common/theme/Theme"
import { ActionDialog } from "../../../components/ActionDialog"
import { ProgressDialog } from "../../../components/ProgressDialog"
import { PromptDialog } from "../../../components/PromptDialog"
import { Toast } from "../../../components/Toast"
import { isRunningInElectron } from "../../helpers/platform"
import { DialogProvider } from "../../hooks/useDialog"
import { ProgressProvider } from "../../hooks/useProgress"
import { PromptProvider } from "../../hooks/usePrompt"
import { StoreContext } from "../../hooks/useStores"
import { ThemeContext } from "../../hooks/useTheme"
import { ToastProvider } from "../../hooks/useToast"
import RootStore from "../../stores/RootStore"
import { GlobalKeyboardShortcut } from "../KeyboardShortcut/GlobalKeyboardShortcut"
import { RootView } from "../RootView/RootView"
import { EmotionThemeProvider } from "../Theme/EmotionThemeProvider"
import { GlobalCSS } from "../Theme/GlobalCSS"
import { ElectronCallbackHandler } from "./ElectronCallbackHandler"
import { LocalizationProvider } from "./LocalizationProvider"

const rootStore = new RootStore()

export function App() {
  return (
    <React.StrictMode>
      <StoreContext.Provider value={rootStore}>
        <ThemeContext.Provider value={defaultTheme}>
          <EmotionThemeProvider>
            <HelmetProvider>
              <ToastProvider component={Toast}>
                <PromptProvider component={PromptDialog}>
                  <DialogProvider component={ActionDialog}>
                    <ProgressProvider component={ProgressDialog}>
                      <LocalizationProvider>
                        <GlobalKeyboardShortcut />
                        <GlobalCSS />
                        {isRunningInElectron() && <ElectronCallbackHandler />}
                        <RootView />
                      </LocalizationProvider>
                    </ProgressProvider>
                  </DialogProvider>
                </PromptProvider>
              </ToastProvider>
            </HelmetProvider>
          </EmotionThemeProvider>
        </ThemeContext.Provider>
      </StoreContext.Provider>
    </React.StrictMode>
  )
}
