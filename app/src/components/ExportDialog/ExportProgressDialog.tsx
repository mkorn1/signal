import { observer } from "mobx-react-lite"
import { FC, useCallback } from "react"
import { useCancelExport } from "../../actions"
import { useStores } from "../../hooks/useStores"
import { Localized } from "../../localize/useLocalization"
import {
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
} from "../Dialog/Dialog"
import { Button } from "../ui/Button"
import { LinearProgress } from "../ui/LinearProgress"

export const ExportProgressDialog: FC = observer(() => {
  const rootStore = useStores()
  const { exportStore } = rootStore
  const cancelExport = useCancelExport()
  const { openExportProgressDialog: open, progress } = exportStore

  const onClickCancel = useCallback(() => {
    exportStore.openExportProgressDialog = false
    cancelExport()
  }, [exportStore, cancelExport])

  return (
    <Dialog open={open} style={{ minWidth: "20rem" }}>
      <DialogTitle>
        <Localized name="exporting-audio" />
      </DialogTitle>
      <DialogContent>
        <LinearProgress value={progress} max={1} />
      </DialogContent>
      <DialogActions>
        <Button onClick={onClickCancel}>
          <Localized name="cancel" />
        </Button>
      </DialogActions>
    </Dialog>
  )
})
