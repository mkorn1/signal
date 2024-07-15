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
  trackIndex: number
}

export const TrackListContextMenu: FC<TrackListContextMenuProps> = observer(
  ({ trackIndex, ...props }) => {
    const rootStore = useStores()
    const { handleClose } = props
    const [isDialogOpened, setDialogOpened] = useState(false)
    const [isColorPickerOpened, setColorPickerOpened] = useState(false)

    const onClickAdd = useCallback(() => addTrack(rootStore)(), [trackIndex])
    const onClickDelete = useCallback(
      () => removeTrack(rootStore)(trackIndex),
      [trackIndex],
    )
    const onClickProperty = () => setDialogOpened(true)
    const onClickChangeTrackColor = () => setColorPickerOpened(true)

    const onPickColor = (color: string | null) => {
      const track = rootStore.song.tracks[trackIndex]
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
          trackIndex={trackIndex}
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
