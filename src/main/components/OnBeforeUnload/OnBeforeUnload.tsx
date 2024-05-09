import { useDialog } from "dialog-hooks"
import { observer } from "mobx-react-lite"
import { useEffect } from "react"
import { useLocalization } from "../../../common/localize/useLocalization"
import { isRunningInElectron } from "../../helpers/platform"
import { useStores } from "../../hooks/useStores"

export const OnBeforeUnload = observer(() => {
  const rootStore = useStores()
  const localized = useLocalization()
  const dialog = useDialog()

  useEffect(() => {
    const listener = (e: BeforeUnloadEvent) => {
      if (!rootStore.song.isSaved) {
        const message = localized(
          "confirm-close",
          "Your edits have not been saved. Be sure to download it before exiting. Do you really want to close it?",
        )
        if (isRunningInElectron()) {
          // do not close the window immediately
          e.returnValue = false
          dialog
            .show({
              title: message,
              actions: [
                {
                  title: localized("close", "Close"),
                  key: true,
                },
                {
                  title: localized("cancel", "Cancel"),
                  key: false,
                },
              ],
            })
            .then((shouldClose) => {
              if (shouldClose) {
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
