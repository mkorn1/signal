import { useStores } from "../../../../hooks/useStores"
import { MouseGesture } from "../NoteMouseHandler"

export const useAddNoteToSelectionGesture = (): MouseGesture<[number]> => {
  const { pianoRollStore } = useStores()

  return {
    onMouseDown(_e: MouseEvent, noteId: number) {
      pianoRollStore.selectedNoteIds.push(noteId)
    },
  }
}
