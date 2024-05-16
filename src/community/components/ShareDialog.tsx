import { CloudSong } from "@signal-app/api"
import { observer } from "mobx-react-lite"
import { FC } from "react"
import {
  Localized,
  useLocalization,
} from "../../common/localize/useLocalization"
import { Button } from "../../components/Button"
import {
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
} from "../../components/Dialog"
import { LinkShare } from "../../components/LinkShare"

export interface ShareDialogProps {
  song: CloudSong
  open: boolean
  onClose: () => void
}

export const ShareDialog: FC<ShareDialogProps> = observer(
  ({ song, open, onClose }) => {
    const localized = useLocalization()

    return (
      <Dialog open={open} onOpenChange={onClose} style={{ minWidth: "20rem" }}>
        <DialogTitle>
          <Localized name="share" />
        </DialogTitle>
        <DialogContent>
          <LinkShare
            url={getCloudSongUrl(song.id)}
            text={`🎶 ${song.name} by ${song.user?.name} from @signalmidi\n#midi #signalmidi`}
          />
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

const getCloudSongUrl = (cloudSongId: string) =>
  `${window.location.origin}/songs/${cloudSongId}`
