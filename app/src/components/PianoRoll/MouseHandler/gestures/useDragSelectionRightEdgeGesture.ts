import { useMoveDraggableGesture } from "./useMoveDraggableGesture"

export const useDragSelectionRightEdgeGesture = () => {
  const moveDraggableAction = useMoveDraggableGesture()

  return {
    onMouseDown(e: MouseEvent, selectedNoteIds: number[]) {
      moveDraggableAction.onMouseDown(
        e,
        {
          type: "selection",
          position: "right",
        },
        selectedNoteIds.map((noteId) => ({
          type: "note",
          position: "right",
          noteId,
        })),
      )
    },
  }
}
