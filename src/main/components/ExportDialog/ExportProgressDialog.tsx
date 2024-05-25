import { observer } from "mobx-react-lite"
import { FC, useCallback } from "react"
import { Button } from "../../../components/Button"
import {
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
} from "../../../components/Dialog"
import { LinearProgress } from "../../../components/LinearProgress"
import { cancelExport } from "../../actions"
import { useStores } from "../../hooks/useStores"
import { Localized } from "../../localize/useLocalization"

export const ExportProgressDialog: FC = observer(() => {
  const rootStore = useStores()
  const { exportStore } = rootStore
  const { openExportProgressDialog: open, progress } = exportStore

  const onClickCancel = useCallback(() => {
    exportStore.openExportProgressDialog = false
    cancelExport(rootStore)()
  }, [])

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
