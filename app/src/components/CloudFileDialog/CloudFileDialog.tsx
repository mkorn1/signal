import { observer } from "mobx-react-lite"
import { useCallback } from "react"
import { useStores } from "../../hooks/useStores"
import { Localized } from "../../localize/useLocalization"
import {
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
} from "../Dialog/Dialog"
import { Button } from "../ui/Button"
import { CloudFileList } from "./CloudFileList"

export const CloudFileDialog = observer(() => {
  const {
    rootViewStore,
    rootViewStore: { openCloudFileDialog },
  } = useStores()

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
