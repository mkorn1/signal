import { useLocalization } from "../../common/localize/useLocalization"
import { createSong, saveSong } from "../actions"
import { openFile, saveFile, saveFileAs } from "../actions/file"
import { useStores } from "./useStores"
import { useToast } from "./useToast"

export const useSongFile = () => {
  const rootStore = useStores()
  const toast = useToast()
  const localized = useLocalization()
  const { song } = rootStore

  return {
    async createNewSong() {
      if (
        song.isSaved ||
        confirm(localized("confirm-new", "Are you sure you want to continue?"))
      ) {
        createSong(rootStore)()
      }
    },
    async openSong() {
      try {
        if (
          song.isSaved ||
          confirm(
            localized("confirm-open", "Are you sure you want to continue?"),
          )
        ) {
          await openFile(rootStore)
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
