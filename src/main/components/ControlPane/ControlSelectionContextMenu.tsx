import React, { FC, useCallback } from "react"
import {
  ContextMenu,
  ContextMenuProps,
  ContextMenuHotKey as HotKey,
} from "../../../components/ContextMenu"
import { MenuItem } from "../../../components/Menu"
import {
  copyControlSelection,
  deleteControlSelection,
  duplicateControlSelection,
  pasteControlSelection,
} from "../../actions/control"
import { useStores } from "../../hooks/useStores"
import { envString } from "../../localize/envString"
import { Localized } from "../../localize/useLocalization"

export const ControlSelectionContextMenu: FC<ContextMenuProps> = React.memo(
  (props) => {
    const { handleClose } = props
    const rootStore = useStores()
    const { controlStore } = rootStore
    const isEventSelected = controlStore.selectedEventIds.length > 0

    const onClickCut = useCallback(() => {
      copyControlSelection(rootStore)()
      deleteControlSelection(rootStore)()
      handleClose()
    }, [])

    const onClickCopy = useCallback(() => {
      copyControlSelection(rootStore)()
      handleClose()
    }, [])

    const onClickPaste = useCallback(() => {
      pasteControlSelection(rootStore)()
      handleClose()
    }, [])

    const onClickDuplicate = useCallback(() => {
      duplicateControlSelection(rootStore)()
      handleClose()
    }, [])

    const onClickDelete = useCallback(() => {
      deleteControlSelection(rootStore)()
      handleClose()
    }, [])

    return (
      <ContextMenu {...props}>
        <MenuItem onClick={onClickCut} disabled={!isEventSelected}>
          <Localized name="cut" />
          <HotKey>{envString.cmdOrCtrl}+X</HotKey>
        </MenuItem>
        <MenuItem onClick={onClickCopy} disabled={!isEventSelected}>
          <Localized name="copy" />
          <HotKey>{envString.cmdOrCtrl}+C</HotKey>
        </MenuItem>
        <MenuItem onClick={onClickPaste}>
          <Localized name="paste" />
          <HotKey>{envString.cmdOrCtrl}+V</HotKey>
        </MenuItem>
        <MenuItem onClick={onClickDuplicate} disabled={!isEventSelected}>
          <Localized name="duplicate" />
          <HotKey>{envString.cmdOrCtrl}+D</HotKey>
        </MenuItem>
        <MenuItem onClick={onClickDelete} disabled={!isEventSelected}>
          <Localized name="delete" />
          <HotKey>Del</HotKey>
        </MenuItem>
      </ContextMenu>
    )
  },
)
