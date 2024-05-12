import { FC } from "react"
import { Button } from "../../../components/Button"
import {
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
} from "../../../components/Dialog"
import { Localized } from "../../../components/Localized"

export interface InitializeErrorDialogProps {
  open: boolean
  message: string
  onClose: () => void
}

export const InitializeErrorDialog: FC<InitializeErrorDialogProps> = ({
  open,
  message,
  onClose,
}) => {
  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogTitle>
        <Localized default="Error occured in launch process">
          initialize-error
        </Localized>
      </DialogTitle>
      <DialogContent>{message}</DialogContent>
      <DialogActions>
        <Button onClick={onClose}>
          <Localized default="Close">close</Localized>
        </Button>
      </DialogActions>
    </Dialog>
  )
}
