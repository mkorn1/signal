import { observer } from "mobx-react-lite"
import { FC } from "react"
import { useStores } from "../../hooks/useStores"
import { Localized } from "../../localize/useLocalization"
import { MenuDivider, MenuItem } from "../ui/Menu"
import {
  useCopySelection,
  useDeleteSelection,
  useDuplicateSelection,
  usePasteSelection,
  useSelectAllNotes,
  useSelectNextNote,
  useSelectPreviousNote,
  useQuantizeSelectedNotes,
  useTransposeSelection,
} from "../../actions"
import { useRedo, useUndo } from "../../actions/history"

export const EditMenu: FC<{ close: () => void }> = observer(({ close }) => {
  const { historyStore, pianoRollStore } = useStores()
  const undo = useUndo()
  const redo = useRedo()
  const copySelection = useCopySelection()
  const pasteSelection = usePasteSelection()
  const deleteSelection = useDeleteSelection()
  const duplicateSelection = useDuplicateSelection()
  const selectAllNotes = useSelectAllNotes()
  const selectNextNote = useSelectNextNote()
  const selectPreviousNote = useSelectPreviousNote()
  const quantizeSelectedNotes = useQuantizeSelectedNotes()
  const transposeSelection = useTransposeSelection()
  const anySelectedNotes = pianoRollStore.selectedNoteIds.length > 0

  const onClickUndo = async () => {
    close()
    await undo()
  }

  const onClickRedo = async () => {
    close()
    await redo()
  }

  const onClickCut = async () => {
    close()
    await copySelection()
    await deleteSelection()
  }

  const onClickCopy = async () => {
    close()
    await copySelection()
  }

  const onClickPaste = async () => {
    close()
    await pasteSelection()
  }

  const onClickDelete = async () => {
    close()
    await deleteSelection()
  }

  const onClickSelectAll = async () => {
    close()
    await selectAllNotes()
  }

  const onClickDuplicate = async () => {
    close()
    await duplicateSelection()
  }

  const onClickSelectNextNote = async () => {
    close()
    await selectNextNote()
  }

  const onClickSelectPreviousNote = async () => {
    close()
    await selectPreviousNote()
  }

  const onClickQuantizeSelectedNotes = async () => {
    close()
    await quantizeSelectedNotes()
  }

  const onClickTransposeUpOctave = async () => {
    close()
    await transposeSelection(12)
  }

  const onClickTransposeDownOctave = async () => {
    close()
    await transposeSelection(-12)
  }

  const onClickTransposeUp = async () => {
    close()
    await transposeSelection(1)
  }

  const onClickTransposeDown = async () => {
    close()
    await transposeSelection(-1)
  }

  return (
    <>
      <MenuItem onClick={onClickUndo} disabled={!historyStore.hasUndo}>
        <Localized name="undo" />
      </MenuItem>

      <MenuItem onClick={onClickRedo} disabled={!historyStore.hasRedo}>
        <Localized name="redo" />
      </MenuItem>

      <MenuDivider />

      <MenuItem onClick={onClickCut} disabled={!anySelectedNotes}>
        <Localized name="cut" />
      </MenuItem>

      <MenuItem onClick={onClickCopy} disabled={!anySelectedNotes}>
        <Localized name="copy" />
      </MenuItem>

      <MenuItem onClick={onClickPaste} disabled={!anySelectedNotes}>
        <Localized name="paste" />
      </MenuItem>

      <MenuItem onClick={onClickDelete} disabled={!anySelectedNotes}>
        <Localized name="delete" />
      </MenuItem>

      <MenuItem onClick={onClickDuplicate} disabled={!anySelectedNotes}>
        <Localized name="duplicate" />
      </MenuItem>

      <MenuDivider />

      <MenuItem onClick={onClickSelectAll}>
        <Localized name="select-all" />
      </MenuItem>

      <MenuItem onClick={onClickSelectNextNote} disabled={!anySelectedNotes}>
        <Localized name="select-next" />
      </MenuItem>

      <MenuItem
        onClick={onClickSelectPreviousNote}
        disabled={!anySelectedNotes}
      >
        <Localized name="select-previous" />
      </MenuItem>

      <MenuDivider />

      <MenuItem
        onClick={onClickQuantizeSelectedNotes}
        disabled={!anySelectedNotes}
      >
        <Localized name="quantize" />
      </MenuItem>

      <MenuItem onClick={onClickTransposeUp} disabled={!anySelectedNotes}>
        <Localized name="transpose-up" />
      </MenuItem>

      <MenuItem onClick={onClickTransposeDown} disabled={!anySelectedNotes}>
        <Localized name="transpose-down" />
      </MenuItem>

      <MenuItem onClick={onClickTransposeUpOctave} disabled={!anySelectedNotes}>
        <Localized name="transpose-up-octave" />
      </MenuItem>

      <MenuItem
        onClick={onClickTransposeDownOctave}
        disabled={!anySelectedNotes}
      >
        <Localized name="transpose-down-octave" />
      </MenuItem>
    </>
  )
})
