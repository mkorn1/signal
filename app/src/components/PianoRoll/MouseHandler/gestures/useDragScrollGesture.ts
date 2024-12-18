import { observeDrag } from "../../../../helpers/observeDrag"
import { useStores } from "../../../../hooks/useStores"
import { MouseGesture } from "../NoteMouseHandler"

export const useDragScrollGesture = (): MouseGesture => {
  const { pianoRollStore } = useStores()
  return {
    onMouseDown(_e: MouseEvent) {
      observeDrag({
        onMouseMove: (e: MouseEvent) => {
          pianoRollStore.scrollBy(e.movementX, e.movementY)
          pianoRollStore.autoScroll = false
        },
      })
    },
  }
}
