import { observer } from "mobx-react-lite"
import { FC } from "react"
import { duplicateTrack, insertTrack, removeTrack } from "../../actions"
import { useStores } from "../../hooks/useStores"
import { Localized } from "../../localize/useLocalization"
import { ContextMenu, ContextMenuProps } from "../ContextMenu/ContextMenu"
import { MenuItem } from "../ui/Menu"

export const ArrangeTrackContextMenu: FC<ContextMenuProps> = observer(
  (props) => {
    const { handleClose } = props
    const rootStore = useStores()
    const {
      song: { tracks },
      arrangeViewStore: { selectedTrackIndex },
    } = rootStore

    return (
      <ContextMenu {...props}>
        <MenuItem
          onClick={(e) => {
            e.stopPropagation()
            insertTrack(rootStore)(selectedTrackIndex + 1)
            handleClose()
          }}
        >
          <Localized name="add-track" />
        </MenuItem>
        {selectedTrackIndex > 0 && tracks.length > 2 && (
          <MenuItem
            onClick={(e) => {
              e.stopPropagation()
              removeTrack(rootStore)(selectedTrackIndex)
              handleClose()
            }}
          >
            <Localized name="delete-track" />
          </MenuItem>
        )}
        {selectedTrackIndex > 0 && (
          <MenuItem
            onClick={(e) => {
              e.stopPropagation()
              duplicateTrack(rootStore)(selectedTrackIndex)
              handleClose()
            }}
          >
            <Localized name="duplicate-track" />
          </MenuItem>
        )}
      </ContextMenu>
    )
  },
)
