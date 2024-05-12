import { observer } from "mobx-react-lite"
import { FC, useEffect, useState } from "react"
import { useLocalization } from "../../../common/localize/useLocalization"
import { setSong } from "../../actions"
import { loadSongFromExternalMidiFile } from "../../actions/cloudSong"
import { songFromArrayBuffer } from "../../actions/file"
import { isRunningInElectron } from "../../helpers/platform"
import { useProgress } from "../../hooks/useProgress"
import { useStores } from "../../hooks/useStores"
import { InitializeErrorDialog } from "./InitializeErrorDialog"

export const OnInit: FC = observer(() => {
  const rootStore = useStores()
  const [isErrorDialogOpen, setIsErrorDialogOpen] = useState(false)
  const [errorMessage, setErrorMessage] = useState("")
  const { show: showProgress } = useProgress()
  const localized = useLocalization()

  const init = async () => {
    const closeProgress = showProgress(
      localized("initializing", "Initializing..."),
    )
    try {
      await rootStore.init()
    } catch (e) {
      setIsErrorDialogOpen(true)
      setErrorMessage((e as Error).message)
    } finally {
      closeProgress()
    }
  }

  const loadExternalMidiIfNeeded = async () => {
    const params = new URLSearchParams(window.location.search)
    const openParam = params.get("open")

    if (openParam) {
      const closeProgress = showProgress(
        localized("loading-external-midi", "Loading external midi file..."),
      )
      try {
        const song = await loadSongFromExternalMidiFile(rootStore)(openParam)
        setSong(rootStore)(song)
      } catch (e) {
        setIsErrorDialogOpen(true)
        setErrorMessage((e as Error).message)
      } finally {
        closeProgress()
      }
    }
  }

  const loadArgumentFileIfNeeded = async () => {
    if (!isRunningInElectron()) {
      return
    }
    const closeProgress = showProgress(
      localized("loading-file", "Loading file..."),
    )
    try {
      const filePath = await window.electronAPI.getArgument()
      if (filePath) {
        const data = await window.electronAPI.readFile(filePath)
        const song = songFromArrayBuffer(data, filePath)
        setSong(rootStore)(song)
      }
    } catch (e) {
      setIsErrorDialogOpen(true)
      setErrorMessage((e as Error).message)
    } finally {
      closeProgress()
    }
  }

  useEffect(() => {
    ;(async () => {
      await init()
      await loadExternalMidiIfNeeded()
      await loadArgumentFileIfNeeded()
    })()
  }, [])

  return (
    <>
      <InitializeErrorDialog
        open={isErrorDialogOpen}
        message={errorMessage}
        onClose={() => setIsErrorDialogOpen(false)}
      />
    </>
  )
})
