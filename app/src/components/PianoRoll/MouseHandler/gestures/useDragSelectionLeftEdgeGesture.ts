import { useMoveDraggableGesture } from "./useMoveDraggableGesture"

export const useDragSelectionLeftEdgeGesture = () => {
  const moveDraggableAction = useMoveDraggableGesture()

  return {
    onMouseDown(e: MouseEvent, selectedNoteIds: number[]) {
      moveDraggableAction.onMouseDown(
        e,
        {
          type: "selection",
          position: "left",
        },
        selectedNoteIds.map((noteId) => ({
          type: "note",
          position: "left",
          noteId,
        })),
      )
    },
  }
}
