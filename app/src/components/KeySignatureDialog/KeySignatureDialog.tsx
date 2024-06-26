import styled from "@emotion/styled"
import { observer } from "mobx-react-lite"
import { FC, useEffect } from "react"
import { useStores } from "../../hooks/useStores"
import { Localized } from "../../localize/useLocalization"
import { Scale, ScaleName, scaleValues } from "../../scale/Scale"
import {
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
} from "../Dialog/Dialog"
import { Button } from "../ui/Button"
import { Label } from "../ui/Label"
import { Select } from "../ui/Select"

export interface KeySignatureDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

const keyNames = Array.from({ length: 12 }, (_, i) => {
  const names = [
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
  ]
  return names[i]
})

const Row = styled.div`
  display: flex;
  flex-direction: row;
`

const Column = styled.div`
  display: flex;
  flex-direction: column;
`

export const KeySignatureDialog: FC<KeySignatureDialogProps> = observer(
  ({ open, onOpenChange }) => {
    const { pianoRollStore } = useStores()
    const onClose = () => onOpenChange(false)

    useEffect(() => {
      if (open) {
      }
    }, [open])

    return (
      <Dialog open={open} onOpenChange={onClose} style={{ minWidth: "20rem" }}>
        <DialogTitle>
          <Localized name="scale" />
        </DialogTitle>
        <DialogContent>
          <Row style={{ gap: "1rem" }}>
            <Column style={{ gap: "0.5rem", minWidth: "5rem" }}>
              <Label>
                <Localized name="key" />
              </Label>
              <Select
                value={pianoRollStore.keySignature?.key}
                onChange={(e) => {
                  const key = parseInt(e.target.value)
                  pianoRollStore.keySignature = {
                    scale: pianoRollStore.keySignature?.scale ?? "major",
                    key,
                  }
                }}
              >
                {keyNames.map((name, i) => (
                  <option key={i} value={i}>
                    {name}
                  </option>
                ))}
              </Select>
            </Column>
            <Column style={{ gap: "0.5rem", minWidth: "5rem" }}>
              <Label>
                <Localized name="scale" />
              </Label>
              <Select
                value={pianoRollStore.keySignature?.scale}
                onChange={(e) => {
                  const scale = e.target.value as Scale
                  pianoRollStore.keySignature = {
                    key: pianoRollStore.keySignature?.key ?? 0,
                    scale: scale,
                  }
                }}
              >
                {scaleValues.map((name, i) => (
                  <option key={i} value={name}>
                    <ScaleName scale={name} />
                  </option>
                ))}
              </Select>
            </Column>
          </Row>
        </DialogContent>
        <DialogActions>
          <Button onClick={onClose}>
            <Localized name="close" />
          </Button>
        </DialogActions>
      </Dialog>
    )
  },
)
