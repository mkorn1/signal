import { useToast } from "dialog-hooks"
import {
  GithubAuthProvider,
  GoogleAuthProvider,
  OAuthProvider,
  signInWithCredential,
} from "firebase/auth"
import { observer } from "mobx-react-lite"
import { FC, useEffect } from "react"
import { FirebaseCredential } from "../../../../electron/src/ipc"
import { ElectronAPI } from "../../../../electron/src/preload"
import { useLocalization } from "../../../common/localize/useLocalization"
import { songToMidi } from "../../../common/midi/midiConversion"
import { auth } from "../../../firebase/firebase"
import { setSong } from "../../actions"
import { songFromArrayBuffer } from "../../actions/file"
import { redo, undo } from "../../actions/history"
import {
  copySelectionGlobal,
  cutSelectionGlobal,
  pasteSelectionGlobal,
} from "../../actions/hotkey"
import { useCloudFile } from "../../hooks/useCloudFile"
import { useSongFile } from "../../hooks/useSongFile"
import { useStores } from "../../hooks/useStores"

declare global {
  interface Window {
    electronAPI: ElectronAPI
  }
}

export const ElectronCallbackHandler: FC = observer(() => {
  const rootStore = useStores()
  const localized = useLocalization()
  const localSongFile = useSongFile()
  const cloudSongFile = useCloudFile()
  const toast = useToast()

  const saveFileAs = async () => {
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

  useEffect(() => {
    const unsubscribes = [
      window.electronAPI.onNewFile(async () => {
        const {
          authStore: { isLoggedIn },
        } = rootStore

        if (isLoggedIn) {
          await cloudSongFile.createNewSong()
        } else {
          await localSongFile.createNewSong()
        }
      }),
      window.electronAPI.onClickOpenFile(async () => {
        const {
          authStore: { isLoggedIn },
        } = rootStore

        if (isLoggedIn) {
          await cloudSongFile.openSong()
        } else {
          const { song } = rootStore
          try {
            if (song.isSaved || confirm(localized["confirm-open"])) {
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
        }
      }),
      window.electronAPI.onOpenFile(async ({ filePath }) => {
        const { song } = rootStore
        try {
          if (song.isSaved || confirm(localized["confirm-open"])) {
            const data = await window.electronAPI.readFile(filePath)
            const song = songFromArrayBuffer(data, filePath)
            setSong(rootStore)(song)
            window.electronAPI.addRecentDocument(filePath)
          }
        } catch (e) {
          alert((e as Error).message)
        }
      }),
      window.electronAPI.onSaveFile(async () => {
        const {
          song,
          authStore: { isLoggedIn },
        } = rootStore

        if (isLoggedIn) {
          await cloudSongFile.saveSong()
        } else {
          try {
            if (song.filepath) {
              const data = songToMidi(rootStore.song).buffer
              await window.electronAPI.saveFile(song.filepath, data)
              song.isSaved = true
            } else {
              await saveFileAs()
            }
          } catch (e) {
            alert((e as Error).message)
          }
        }
      }),
      window.electronAPI.onSaveFileAs(async () => {
        const {
          authStore: { isLoggedIn },
        } = rootStore

        if (isLoggedIn) {
          await cloudSongFile.saveAsSong()
        } else {
          await saveFileAs()
        }
      }),
      window.electronAPI.onRename(async () => {
        await cloudSongFile.renameSong()
      }),
      window.electronAPI.onImport(async () => {
        await cloudSongFile.importSong()
      }),
      window.electronAPI.onExportWav(() => {
        rootStore.exportStore.openExportDialog = true
      }),
      window.electronAPI.onUndo(() => {
        undo(rootStore)()
      }),
      window.electronAPI.onRedo(() => {
        redo(rootStore)()
      }),
      window.electronAPI.onCut(() => {
        cutSelectionGlobal(rootStore)()
      }),
      window.electronAPI.onCopy(() => {
        copySelectionGlobal(rootStore)()
      }),
      window.electronAPI.onPaste(() => {
        pasteSelectionGlobal(rootStore)()
      }),
      window.electronAPI.onOpenSetting(() => {
        rootStore.rootViewStore.openSettingDialog = true
      }),
      window.electronAPI.onOpenHelp(() => {
        rootStore.rootViewStore.openHelp = true
      }),
      window.electronAPI.onBrowserSignInCompleted(
        async ({ credential: credentialJSON }) => {
          const credential = createCredential(credentialJSON)
          try {
            await signInWithCredential(auth, credential)
          } catch (e) {
            toast.error("Failed to sign in with Google")
          }
        },
      ),
    ]
    window.electronAPI.ready()
    return () => {
      unsubscribes.forEach((unsubscribe) => unsubscribe())
    }
  }, [])

  return <></>
})

function createCredential(credential: FirebaseCredential) {
  switch (credential.providerId) {
    case "google.com":
      return GoogleAuthProvider.credential(
        credential.idToken,
        credential.accessToken,
      )
    case "github.com":
      return GithubAuthProvider.credential(credential.accessToken)
    case "apple.com":
      let provider = new OAuthProvider("apple.com")
      return provider.credential({
        idToken: credential.idToken,
        accessToken: credential.accessToken,
      })
    default:
      throw new Error("Invalid provider")
  }
}
