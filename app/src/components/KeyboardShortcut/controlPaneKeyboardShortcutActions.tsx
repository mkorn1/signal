import {
  useCopyControlSelection,
  useDeleteControlSelection,
  useDuplicateControlSelection,
  useResetControlSelection,
} from "../../actions/control"

export const useControlPaneKeyboardShortcutActions = () => {
  const resetControlSelection = useResetControlSelection()
  const deleteControlSelection = useDeleteControlSelection()
  const copyControlSelection = useCopyControlSelection()
  const duplicateControlSelection = useDuplicateControlSelection()

  return () => [
    { code: "Escape", run: () => resetControlSelection() },
    { code: "Backspace", run: () => deleteControlSelection() },
    { code: "Delete", run: () => deleteControlSelection() },
    {
      code: "KeyC",
      metaKey: true,
      run: () => copyControlSelection(),
    },
    {
      code: "KeyX",
      metaKey: true,
      run: () => {
        copyControlSelection()
        deleteControlSelection()
      },
    },
    {
      code: "KeyD",
      metaKey: true,
      run: () => duplicateControlSelection(),
    },
  ]
}
