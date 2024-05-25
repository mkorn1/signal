import { observer } from "mobx-react-lite"
import { useCallback } from "react"
import { Button } from "../../../components/Button"
import {
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
} from "../../../components/Dialog"
import { useStores } from "../../hooks/useStores"
import { Localized } from "../../localize/useLocalization"
import { CloudFileList } from "./CloudFileList"

export const CloudFileDialog = observer(() => {
  const rootStore = useStores()
  const {
    rootViewStore,
    rootViewStore: { openCloudFileDialog },
  } = rootStore

  const onClose = useCallback(
    () => (rootViewStore.openCloudFileDialog = false),
    [rootViewStore],
  )

  return (
    <Dialog
      open={openCloudFileDialog}
      onOpenChange={onClose}
      style={{ minWidth: "30rem" }}
    >
      <DialogTitle>
        <Localized name="files" />
      </DialogTitle>
      <DialogContent>
        <CloudFileList />
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>
          <Localized name="close" />
        </Button>
      </DialogActions>
    </Dialog>
  )
})
