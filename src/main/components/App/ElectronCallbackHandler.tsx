import { GoogleAuthProvider, signInWithCredential } from "firebase/auth"
import { FC, useEffect } from "react"
import { ElectronAPI } from "../../../../electron/src/preload"
import { useLocalization } from "../../../common/localize/useLocalization"
import { songToMidi } from "../../../common/midi/midiConversion"
import { auth } from "../../../firebase/firebase"
import { createSong, setSong } from "../../actions"
import { songFromArrayBuffer } from "../../actions/file"
import { redo, undo } from "../../actions/history"
import {
  copySelectionGlobal,
  cutSelectionGlobal,
  pasteSelectionGlobal,
} from "../../actions/hotkey"
import { useStores } from "../../hooks/useStores"
import RootStore from "../../stores/RootStore"

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
    window.electronAPI.addRecentDocument(path)
  } catch (e) {
    alert((e as Error).message)
  }
}

export const ElectronCallbackHandler: FC = () => {
  const rootStore = useStores()
  const localized = useLocalization()

  useEffect(() => {
    window.electronAPI.onNewFile(() => {
      const { song } = rootStore
      if (
        song.isSaved ||
        confirm(localized("confirm-new", "Are you sure you want to continue?"))
      ) {
        createSong(rootStore)()
      }
    })
    window.electronAPI.onClickOpenFile(async () => {
      const { song } = rootStore
      try {
        if (
          song.isSaved ||
          confirm(
            localized("confirm-open", "Are you sure you want to continue?"),
          )
        ) {
          const res = await window.electronAPI.showOpenDialog()
          if (res === null) {
            return // canceled
          }
          const { path, content } = res
          const song = songFromArrayBuffer(content, path)
          setSong(rootStore)(song)
          window.electronAPI.addRecentDocument(path)
        }
      } catch (e) {
        alert((e as Error).message)
      }
    })
    window.electronAPI.onOpenFile(async ({ filePath }) => {
      const { song } = rootStore
      try {
        if (
          song.isSaved ||
          confirm(
            localized("confirm-open", "Are you sure you want to continue?"),
          )
        ) {
          const data = await window.electronAPI.readFile(filePath)
          const song = songFromArrayBuffer(data, filePath)
          setSong(rootStore)(song)
          window.electronAPI.addRecentDocument(filePath)
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
    window.electronAPI.onRename(async () => {})
    window.electronAPI.onImport(async () => {})
    window.electronAPI.onExportWav(() => {
      rootStore.exportStore.openExportDialog = true
    })
    window.electronAPI.onUndo(() => {
      undo(rootStore)()
    })
    window.electronAPI.onRedo(() => {
      redo(rootStore)()
    })
    window.electronAPI.onCut(() => {
      cutSelectionGlobal(rootStore)()
    })
    window.electronAPI.onCopy(() => {
      copySelectionGlobal(rootStore)()
    })
    window.electronAPI.onPaste(() => {
      pasteSelectionGlobal(rootStore)()
    })
    window.electronAPI.onOpenSetting(() => {
      rootStore.rootViewStore.openSettingDialog = true
    })
    window.electronAPI.onOpenHelp(() => {
      rootStore.rootViewStore.openHelp = true
    })
    window.electronAPI.onIdTokenReceived(async ({ idToken }) => {
      const credential = GoogleAuthProvider.credential(idToken)
      await signInWithCredential(auth, credential)
    })
  }, [])

  return <></>
}
