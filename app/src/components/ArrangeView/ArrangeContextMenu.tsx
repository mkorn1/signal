import { observer } from "mobx-react-lite"
import { FC, useCallback } from "react"
import {
  arrangeCopySelection,
  arrangeDeleteSelection,
  arrangeDuplicateSelection,
  arrangePasteSelection,
  arrangeTransposeSelection,
} from "../../actions/arrangeView"
import { useStores } from "../../hooks/useStores"
import { envString } from "../../localize/envString"
import { Localized } from "../../localize/useLocalization"
import {
  ContextMenu,
  ContextMenuProps,
  ContextMenuHotKey as HotKey,
} from "../ContextMenu/ContextMenu"
import { MenuDivider, MenuItem } from "../ui/Menu"

export const ArrangeContextMenu: FC<ContextMenuProps> = observer((props) => {
  const { handleClose } = props
  const rootStore = useStores()
  const { arrangeViewStore } = rootStore
  const isNoteSelected = Object.values(arrangeViewStore.selectedEventIds).some(
    (e) => e.length > 0,
  )

  const onClickVelocity = useCallback(() => {
    arrangeViewStore.openVelocityDialog = true
    handleClose()
  }, [arrangeViewStore])

  return (
    <ContextMenu {...props}>
      <MenuItem
        onClick={(e) => {
          e.stopPropagation()
          handleClose()
          arrangeCopySelection(rootStore)()
          arrangeDeleteSelection(rootStore)()
        }}
        disabled={!isNoteSelected}
      >
        <Localized name="cut" />
        <HotKey>{envString.cmdOrCtrl}+X</HotKey>
      </MenuItem>
      <MenuItem
        onClick={(e) => {
          e.stopPropagation()
          handleClose()
          arrangeCopySelection(rootStore)()
        }}
        disabled={!isNoteSelected}
      >
        <Localized name="copy" />
        <HotKey>{envString.cmdOrCtrl}+C</HotKey>
      </MenuItem>
      <MenuItem
        onClick={(e) => {
          e.stopPropagation()
          handleClose()
          arrangePasteSelection(rootStore)()
        }}
      >
        <Localized name="paste" />
        <HotKey>{envString.cmdOrCtrl}+V</HotKey>
      </MenuItem>
      <MenuItem
        onClick={(e) => {
          e.stopPropagation()
          handleClose()
          arrangeDuplicateSelection(rootStore)()
        }}
        disabled={arrangeViewStore.selection === null}
      >
        <Localized name="duplicate" />
        <HotKey>{envString.cmdOrCtrl}+D</HotKey>
      </MenuItem>
      <MenuItem
        onClick={(e) => {
          e.stopPropagation()
          handleClose()
          arrangeDeleteSelection(rootStore)()
        }}
        disabled={!isNoteSelected}
      >
        <Localized name="delete" />
        <HotKey>Del</HotKey>
      </MenuItem>
      <MenuDivider />
      <MenuItem
        onClick={(e) => {
          e.stopPropagation()
          handleClose()
          arrangeTransposeSelection(rootStore)(12)
        }}
        disabled={!isNoteSelected}
      >
        <Localized name="one-octave-up" />
      </MenuItem>
      <MenuItem
        onClick={(e) => {
          e.stopPropagation()
          handleClose()
          arrangeTransposeSelection(rootStore)(-12)
        }}
        disabled={!isNoteSelected}
      >
        <Localized name="one-octave-down" />
      </MenuItem>
      <MenuItem
        onClick={(e) => {
          e.stopPropagation()
          handleClose()
          arrangeViewStore.openTransposeDialog = true
        }}
        disabled={!isNoteSelected}
      >
        <Localized name="transpose" />
        <HotKey>T</HotKey>
      </MenuItem>
      <MenuItem onClick={onClickVelocity} disabled={!isNoteSelected}>
        <Localized name="velocity" />
      </MenuItem>
    </ContextMenu>
  )
})
