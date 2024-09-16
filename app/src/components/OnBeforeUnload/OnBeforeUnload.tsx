import { observer } from "mobx-react-lite"
import { useEffect } from "react"
import { isRunningInElectron } from "../../helpers/platform"
import { useStores } from "../../hooks/useStores"
import { useLocalization } from "../../localize/useLocalization"

export const OnBeforeUnload = observer(() => {
  const rootStore = useStores()
  const localized = useLocalization()

  useEffect(() => {
    const listener = (e: BeforeUnloadEvent) => {
      if (!rootStore.song.isSaved) {
        const message = localized["confirm-close"]
        if (isRunningInElectron()) {
          // do not close the window immediately
          e.returnValue = false
          e.preventDefault()
          window.electronAPI
            .showMessageBox({
              type: "question",
              message,
              buttons: [localized["close"], localized["cancel"]],
            })
            .then((button) => {
              if (button === 0) {
                window.electronAPI.closeMainWindow()
              }
            })
        } else {
          e.returnValue = message
        }
      }
    }
    window.addEventListener("beforeunload", listener)

    return () => {
      window.removeEventListener("beforeunload", listener)
    }
  }, [])
  return <></>
})
