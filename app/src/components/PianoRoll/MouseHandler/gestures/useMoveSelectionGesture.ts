import { useCloneSelection } from "../../../../actions"
import { useMoveDraggableGesture } from "./useMoveDraggableGesture"

export const useMoveSelectionGesture = () => {
  const moveDraggableAction = useMoveDraggableGesture()
  const cloneSelection = useCloneSelection()

  return {
    onMouseDown(e: MouseEvent, selectedNoteIds: number[]) {
      const isCopy = e.metaKey

      if (isCopy) {
        cloneSelection()
      }

      return moveDraggableAction.onMouseDown(
        e,
        { type: "selection", position: "center" },
        selectedNoteIds.map((noteId) => ({
          type: "note",
          position: "center",
          noteId,
        })),
      )
    },
  }
}
