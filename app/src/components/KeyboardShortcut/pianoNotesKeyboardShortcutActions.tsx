import {
  copySelection,
  deleteSelection,
  duplicateSelection,
  pasteSelection,
  quantizeSelectedNotes,
  resetSelection,
  selectAllNotes,
  transposeSelection,
  useSelectNextNote,
  useSelectPreviousNote,
} from "../../actions"
import { useStores } from "../../hooks/useStores"

export const usePianoNotesKeyboardShortcutActions = () => {
  const rootStore = useStores()
  const selectNextNote = useSelectNextNote()
  const selectPreviousNote = useSelectPreviousNote()

  return () => [
    {
      code: "KeyC",
      metaKey: true,
      run: () => copySelection(rootStore)(),
    },
    {
      code: "KeyX",
      metaKey: true,
      run: () => {
        copySelection(rootStore)()
        deleteSelection(rootStore)()
      },
    },
    {
      code: "KeyV",
      metaKey: true,
      run: () => pasteSelection(rootStore)(),
    },
    {
      code: "KeyD",
      metaKey: true,
      run: () => duplicateSelection(rootStore)(),
    },
    {
      code: "KeyA",
      metaKey: true,
      run: () => selectAllNotes(rootStore)(),
    },
    {
      code: "KeyQ",
      run: () => quantizeSelectedNotes(rootStore)(),
    },
    {
      code: "KeyT",
      run: () => (rootStore.pianoRollStore.openTransposeDialog = true),
    },
    { code: "Delete", run: () => deleteSelection(rootStore)() },
    {
      code: "Backspace",
      run: () => deleteSelection(rootStore)(),
    },
    {
      code: "ArrowUp",
      shiftKey: true,
      run: () => transposeSelection(rootStore)(12),
    },
    {
      code: "ArrowUp",
      run: () => transposeSelection(rootStore)(1),
    },
    {
      code: "ArrowDown",
      shiftKey: true,
      run: () => transposeSelection(rootStore)(-12),
    },
    {
      code: "ArrowDown",
      run: () => transposeSelection(rootStore)(-1),
    },
    {
      code: "ArrowRight",
      run: selectNextNote,
      enabled: () => rootStore.pianoRollStore.mouseMode == "pencil",
    },
    {
      code: "ArrowLeft",
      run: selectPreviousNote,
      enabled: () => rootStore.pianoRollStore.mouseMode == "pencil",
    },
    { code: "Escape", run: () => resetSelection(rootStore)() },
  ]
}
