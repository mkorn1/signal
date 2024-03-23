import type { ElectronAPI } from "../../../electron/src/preload"
import { localized } from "../../common/localize/localizedString"
import { songToMidi } from "../../common/midi/midiConversion"
import { createSong, setSong } from "../actions"
import { songFromNativeFile } from "../actions/file"
import RootStore from "./RootStore"

declare global {
  interface Window {
    electronAPI: ElectronAPI
  }
}

const saveFileAs = async (rootStore: RootStore) => {
  const { song } = rootStore
  try {
    const res = await window.electronAPI.showSaveDialog()
    if (res === null) {
      return // canceled
    }
    const { path } = res
    const data = songToMidi(song).buffer
    song.filepath = path
    song.isSaved = true
    await window.electronAPI.saveFile(path, data)
  } catch (e) {
    alert((e as Error).message)
  }
}

export const registerElectronReactions = (rootStore: RootStore) => {
  window.electronAPI.onNewFile(() => {
    const { song } = rootStore
    if (
      song.isSaved ||
      confirm(localized("confirm-new", "Are you sure you want to continue?"))
    ) {
      createSong(rootStore)()
    }
  })
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
  window.electronAPI.onSaveFile(async () => {
    const { song } = rootStore
    try {
      if (song.filepath) {
        const data = songToMidi(rootStore.song).buffer
        await window.electronAPI.saveFile(song.filepath, data)
      } else {
        await saveFileAs(rootStore)
      }
    } catch (e) {
      alert((e as Error).message)
    }
  })
  window.electronAPI.onSaveFileAs(async () => {
    await saveFileAs(rootStore)
  })
  window.electronAPI.onOpenSetting(() => {
    rootStore.rootViewStore.openSettingDialog = true
  })
  window.electronAPI.onOpenHelp(() => {
    rootStore.rootViewStore.openHelp = true
  })
}
