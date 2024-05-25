import { FC } from "react"
import { ContextMenu, ContextMenuProps } from "../../../components/ContextMenu"
import { MenuItem } from "../../../components/Menu"
import { Localized } from "../../localize/useLocalization"

export type TrackListContextMenuProps = ContextMenuProps & {
  onClickAdd: () => void
  onClickDelete: () => void
  onClickProperty: () => void
  onClickChangeTrackColor: () => void
}

export const TrackListContextMenu: FC<TrackListContextMenuProps> = ({
  onClickAdd,
  onClickDelete,
  onClickProperty,
  onClickChangeTrackColor,
  ...props
}) => {
  const { handleClose } = props
  return (
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
  )
}
