import Color from "color"
import { observer } from "mobx-react-lite"
import { FC, useCallback, useState } from "react"
import { addTrack, removeTrack } from "../../actions"
import { useStores } from "../../hooks/useStores"
import { Localized } from "../../localize/useLocalization"
import { ColorPicker } from "../ColorPicker/ColorPicker"
import { ContextMenu, ContextMenuProps } from "../ContextMenu/ContextMenu"
import { MenuItem } from "../ui/Menu"
import { TrackDialog } from "./TrackDialog"

export interface TrackListContextMenuProps extends ContextMenuProps {
  trackId: number
}

export const TrackListContextMenu: FC<TrackListContextMenuProps> = observer(
  ({ trackId, ...props }) => {
    const rootStore = useStores()
    const { handleClose } = props
    const [isDialogOpened, setDialogOpened] = useState(false)
    const [isColorPickerOpened, setColorPickerOpened] = useState(false)

    const onClickAdd = useCallback(() => addTrack(rootStore)(), [trackId])
    const onClickDelete = useCallback(
      () => removeTrack(rootStore)(trackId),
      [trackId],
    )
    const onClickProperty = () => setDialogOpened(true)
    const onClickChangeTrackColor = () => setColorPickerOpened(true)

    const onPickColor = (color: string | null) => {
      const track = rootStore.song.tracks[trackId]
      if (color === null) {
        track.setColor(null)
        return
      }
      const obj = Color(color)
      track.setColor({
        red: Math.floor(obj.red()),
        green: Math.floor(obj.green()),
        blue: Math.floor(obj.blue()),
        alpha: 0xff,
      })
    }

    return (
      <>
        <ContextMenu {...props}>
          <MenuItem
            onClick={(e) => {
              e.stopPropagation()
              onClickAdd()
              handleClose()
            }}
          >
            <Localized name="add-track" />
          </MenuItem>
          <MenuItem
            onClick={(e) => {
              e.stopPropagation()
              onClickDelete()
              handleClose()
            }}
          >
            <Localized name="delete-track" />
          </MenuItem>
          <MenuItem
            onClick={(e) => {
              e.stopPropagation()
              onClickProperty()
              handleClose()
            }}
          >
            <Localized name="property" />
          </MenuItem>
          <MenuItem
            onClick={(e) => {
              e.stopPropagation()
              onClickChangeTrackColor()
              handleClose()
            }}
          >
            <Localized name="change-track-color" />
          </MenuItem>
        </ContextMenu>
        <TrackDialog
          trackId={trackId}
          open={isDialogOpened}
          onClose={() => setDialogOpened(false)}
        />
        <ColorPicker
          open={isColorPickerOpened}
          onSelect={onPickColor}
          onClose={() => setColorPickerOpened(false)}
        />
      </>
    )
  },
)
