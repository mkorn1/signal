import { observer } from "mobx-react-lite"
import { FC, useCallback, useEffect, useState } from "react"
import { canExport, exportSongAsWav } from "../../actions"
import { useStores } from "../../hooks/useStores"
import { Localized } from "../../localize/useLocalization"
import {
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
} from "../Dialog/Dialog"
import { Alert } from "../ui/Alert"
import { Button, PrimaryButton } from "../ui/Button"

export const ExportDialog: FC = observer(() => {
  const rootStore = useStores()
  const { exportStore, song } = rootStore
  const { openExportDialog: open } = exportStore
  const onClose = useCallback(
    () => (exportStore.openExportDialog = false),
    [exportStore],
  )

  const onClickExport = useCallback(() => {
    exportStore.openExportDialog = false
    exportSongAsWav(rootStore)()
  }, [rootStore, exportStore])

  const [exportEnabled, setExportEnabled] = useState(false)
  useEffect(() => {
    if (open) {
      setExportEnabled(canExport(song))
    }
  }, [open])

  return (
    <Dialog open={open} onOpenChange={onClose} style={{ minWidth: "20rem" }}>
      <DialogTitle>
        <Localized name="export-audio" />
      </DialogTitle>
      <DialogContent>
        <p>
          <Localized name="file-type" />: WAV
        </p>
        {!exportEnabled && (
          <Alert severity="warning">
            <Localized name="export-error-too-short" />
          </Alert>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>
          <Localized name="close" />
        </Button>
        {exportEnabled && (
          <PrimaryButton onClick={onClickExport}>
            <Localized name="export" />
          </PrimaryButton>
        )}
      </DialogActions>
    </Dialog>
  )
})
