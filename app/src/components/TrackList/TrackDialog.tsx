import { range } from "lodash"
import { FC, useEffect, useState } from "react"
import { useStores } from "../../hooks/useStores"
import { Localized } from "../../localize/useLocalization"
import {
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
} from "../Dialog/Dialog"
import { Button, PrimaryButton } from "../ui/Button"
import { Label } from "../ui/Label"
import { Select } from "../ui/Select"
import { TextField } from "../ui/TextField"
import { TrackName } from "./TrackName"

export interface TrackDialogProps {
  trackIndex: number
  open: boolean
  onClose: () => void
}

export const TrackDialog: FC<TrackDialogProps> = ({
  trackIndex,
  open,
  onClose,
}) => {
  const { song } = useStores()
  const track = song.tracks[trackIndex]

  const [name, setName] = useState(track.name)
  const [channel, setChannel] = useState(track.channel)

  useEffect(() => {
    setName(track.name)
    setChannel(track.channel)
  }, [trackIndex])

  return (
    <Dialog open={open} onOpenChange={onClose} style={{ minWidth: "20rem" }}>
      <DialogTitle>
        <Localized name="track" />: <TrackName track={track} />
      </DialogTitle>
      <DialogContent>
        <Label>
          <Localized name="track-name" />
        </Label>
        <TextField
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value as string)}
          style={{ width: "100%", marginBottom: "1rem" }}
        />
        <Label>
          <Localized name="channel" />
        </Label>
        <Select
          value={channel}
          onChange={(e) => setChannel(parseInt(e.target.value as string))}
        >
          {range(0, 16).map((v) => (
            <option key={v} value={v.toString()}>
              {v + 1}
              {v === 9 ? (
                <>
                  {" "}
                  (<Localized name="rhythm-track" />)
                </>
              ) : (
                ""
              )}
            </option>
          ))}
        </Select>
      </DialogContent>
      <DialogActions>
        <Button autoFocus onClick={onClose}>
          <Localized name="cancel" />
        </Button>
        <PrimaryButton
          onClick={() => {
            track.channel = channel
            track.setName(name ?? "")
            onClose()
          }}
        >
          <Localized name="ok" />
        </PrimaryButton>
      </DialogActions>
    </Dialog>
  )
}
