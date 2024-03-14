import type { ElectronAPI } from "../../../electron/src/preload.js"
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
}
