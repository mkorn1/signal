import { useStores } from "../../../../hooks/useStores"
import { MouseGesture } from "../NoteMouseHandler"

export const useChangeToolGesture = (): MouseGesture => {
  const { pianoRollStore } = useStores()
  return {
    onMouseDown(_e: MouseEvent) {
      pianoRollStore.toggleTool()
      pianoRollStore.notesCursor = "crosshair"
    },
  }
}
