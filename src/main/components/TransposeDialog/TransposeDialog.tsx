import { FC, useEffect, useState } from "react"
import { Localized } from "../../../common/localize/useLocalization"
import { Button, PrimaryButton } from "../../../components/Button"
import {
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
} from "../../../components/Dialog"
import { TextField } from "../../../components/TextField"

export interface TransposeDialogProps {
  open: boolean
  onClickOK: (value: number) => void
  onClose: () => void
}

export const TransposeDialog: FC<TransposeDialogProps> = ({
  open,
  onClickOK,
  onClose,
}) => {
  const [input, setInput] = useState("0")

  // reset on open
  useEffect(() => {
    if (open) {
      setInput("")
    }
  }, [open])

  const _onClickOK = () => {
    const value = parseInt(input)
    onClickOK(isNaN(value) ? 0 : value)
    onClose()
  }

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogTitle>
        <Localized name="transpose" />
      </DialogTitle>
      <DialogContent>
        <TextField
          type="number"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          autoFocus={true}
          placeholder={"0"}
          onKeyPress={(e) => {
            if (e.key === "Enter") {
              _onClickOK()
            }
          }}
        />
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>
          <Localized name="close" />
        </Button>
        <PrimaryButton onClick={_onClickOK}>
          <Localized name="ok" />
        </PrimaryButton>
      </DialogActions>
    </Dialog>
  )
}
