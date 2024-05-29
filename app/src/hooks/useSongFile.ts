import { useToast } from "dialog-hooks"
import { ChangeEvent } from "react"
import { createSong, openSong, saveSong } from "../actions"
import { openFile, saveFile, saveFileAs } from "../actions/file"
import { useLocalization } from "../localize/useLocalization"
import { useStores } from "./useStores"

export const useSongFile = () => {
  const rootStore = useStores()
  const toast = useToast()
  const localized = useLocalization()
  const { song } = rootStore

  return {
    async createNewSong() {
      if (song.isSaved || confirm(localized["confirm-new"])) {
        createSong(rootStore)()
      }
    },
    async openSong() {
      try {
        if (song.isSaved || confirm(localized["confirm-open"])) {
          await openFile(rootStore)
        }
      } catch (e) {
        toast.error((e as Error).message)
      }
    },
    async openSongLegacy(e: ChangeEvent<HTMLInputElement>) {
      try {
        if (song.isSaved || confirm(localized["confirm-new"])) {
          await openSong(rootStore)(e.currentTarget)
        }
      } catch (e) {
        toast.error((e as Error).message)
      }
    },
    async saveSong() {
      await saveFile(rootStore)
    },
    async saveAsSong() {
      await saveFileAs(rootStore)
    },
    async downloadSong() {
      saveSong(rootStore)()
    },
  }
}
