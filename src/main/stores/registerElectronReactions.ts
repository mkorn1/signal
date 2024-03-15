import type { ElectronAPI } from "../../../electron/src/preload"
import { localized } from "../../common/localize/localizedString"
import { setSong } from "../actions"
import { songFromNativeFile } from "../actions/file"
import RootStore from "./RootStore"

declare global {
  interface Window {
    electronAPI: ElectronAPI
  }
}

export const registerElectronReactions = (rootStore: RootStore) => {
  window.electronAPI.onOpenFile(async () => {
    const { song } = rootStore
    try {
      if (
        song.isSaved ||
        confirm(localized("confirm-open", "Are you sure you want to continue?"))
      ) {
        const res = await window.electronAPI.showOpenDialog()
        if (res === null) {
          return // canceled
        }
        const { path, content } = res
        const song = songFromNativeFile(path, content)
        setSong(rootStore)(song)
      }
    } catch (e) {
      alert((e as Error).message)
    }
  })
  window.electronAPI.onOpenSetting(() => {
    rootStore.rootViewStore.openSettingDialog = true
  })
  window.electronAPI.onOpenHelp(() => {
    rootStore.rootViewStore.openHelp = true
  })
}
