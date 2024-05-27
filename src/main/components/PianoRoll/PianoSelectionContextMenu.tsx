import React, { FC, useCallback } from "react"
import { MenuDivider, MenuItem } from "../../../components/Menu"
import {
  copySelection,
  deleteSelection,
  duplicateSelection,
  pasteSelection,
  quantizeSelectedNotes,
  transposeSelection,
} from "../../actions"
import { useStores } from "../../hooks/useStores"
import { envString } from "../../localize/envString"
import { Localized } from "../../localize/useLocalization"
import {
  ContextMenu,
  ContextMenuProps,
  ContextMenuHotKey as HotKey,
} from "../ContextMenu/ContextMenu"

export const PianoSelectionContextMenu: FC<ContextMenuProps> = React.memo(
  (props) => {
    const { handleClose } = props
    const rootStore = useStores()
    const { pianoRollStore } = rootStore
    const isNoteSelected = pianoRollStore.selectedNoteIds.length > 0

    const onClickCut = useCallback(() => {
      copySelection(rootStore)()
      deleteSelection(rootStore)()
      handleClose()
    }, [])

    const onClickCopy = useCallback(() => {
      copySelection(rootStore)()
      handleClose()
    }, [])

    const onClickPaste = useCallback(() => {
      pasteSelection(rootStore)()
      handleClose()
    }, [])

    const onClickDuplicate = useCallback(() => {
      duplicateSelection(rootStore)()
      handleClose()
    }, [])

    const onClickDelete = useCallback(() => {
      deleteSelection(rootStore)()
      handleClose()
    }, [])

    const onClickOctaveUp = useCallback(() => {
      transposeSelection(rootStore)(12)
      handleClose()
    }, [])

    const onClickOctaveDown = useCallback(() => {
      transposeSelection(rootStore)(-12)
      handleClose()
    }, [])

    const onClickQuantize = useCallback(() => {
      quantizeSelectedNotes(rootStore)()
      handleClose()
    }, [])

    const onClickTranspose = useCallback(() => {
      pianoRollStore.openTransposeDialog = true
      handleClose()
    }, [])

    return (
      <ContextMenu {...props}>
        <MenuItem onClick={onClickCut} disabled={!isNoteSelected}>
          <Localized name="cut" />
          <HotKey>{envString.cmdOrCtrl}+X</HotKey>
        </MenuItem>
        <MenuItem onClick={onClickCopy} disabled={!isNoteSelected}>
          <Localized name="copy" />
          <HotKey>{envString.cmdOrCtrl}+C</HotKey>
        </MenuItem>
        <MenuItem onClick={onClickPaste}>
          <Localized name="paste" />
          <HotKey>{envString.cmdOrCtrl}+V</HotKey>
        </MenuItem>
        <MenuItem onClick={onClickDuplicate} disabled={!isNoteSelected}>
          <Localized name="duplicate" />
          <HotKey>{envString.cmdOrCtrl}+D</HotKey>
        </MenuItem>
        <MenuItem onClick={onClickDelete} disabled={!isNoteSelected}>
          <Localized name="delete" />
          <HotKey>Del</HotKey>
        </MenuItem>
        <MenuDivider />
        <MenuItem onClick={onClickOctaveUp} disabled={!isNoteSelected}>
          <Localized name="one-octave-up" />
          <HotKey>Shift+↑</HotKey>
        </MenuItem>
        <MenuItem onClick={onClickOctaveDown} disabled={!isNoteSelected}>
          <Localized name="one-octave-down" />
          <HotKey>Shift+↓</HotKey>
        </MenuItem>
        <MenuItem onClick={onClickTranspose} disabled={!isNoteSelected}>
          <Localized name="transpose" />
          <HotKey>T</HotKey>
        </MenuItem>
        <MenuDivider />
        <MenuItem onClick={onClickQuantize} disabled={!isNoteSelected}>
          <Localized name="quantize" />
          <HotKey>Q</HotKey>
        </MenuItem>
      </ContextMenu>
    )
  },
)
