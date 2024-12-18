import { MouseGesture } from "../../../../gesture/MouseGesture"
import { useStores } from "../../../../hooks/useStores"

export const useAddNoteToSelectionGesture = (): MouseGesture<[number]> => {
  const { pianoRollStore } = useStores()

  return {
    onMouseDown(_e: MouseEvent, noteId: number) {
      pianoRollStore.selectedNoteIds.push(noteId)
    },
  }
}
