import type { ElectronAPI } from "../../../electron/src/preload"
import RootStore from "./RootStore"

declare global {
  interface Window {
    electronAPI: ElectronAPI
  }
}

export const registerElectronReactions = (rootStore: RootStore) => {
  window.electronAPI.onOpenSetting(() => {
    rootStore.rootViewStore.openSettingDialog = true
  })
  window.electronAPI.onOpenHelp(() => {
    rootStore.rootViewStore.openHelp = true
  })
}
