import { NotePoint } from "../../../entities/transform/NotePoint"
import { observeDrag } from "../../../helpers/observeDrag"
import { PianoRollDraggable } from "../../../stores/PianoRollStore"
import { MouseGesture } from "./NoteMouseHandler"
import { MIN_LENGTH } from "./SelectionMouseHandler"

export const moveDraggableAction =
  (draggable: PianoRollDraggable): MouseGesture =>
  (rootStore) =>
  (e) => {
    const {
      pianoRollStore,
      pianoRollStore: { isQuantizeEnabled, quantizer },
      pushHistory,
    } = rootStore

    const draggablePosition = pianoRollStore.getDraggablePosition(draggable)

    if (draggablePosition === null) {
      return
    }

    let isChanged = false

    const local = rootStore.pianoRollStore.getLocal(e)
    const notePoint = pianoRollStore.transform.getNotePoint(local)
    const offset = NotePoint.sub(draggablePosition, notePoint)

    observeDrag({
      onMouseMove: (e2) => {
        const { selection } = pianoRollStore

        if (selection === null) {
          return
        }

        const quantize = !e2.shiftKey && isQuantizeEnabled
        const minLength = quantize ? quantizer.unit * 2 : MIN_LENGTH
        const local = rootStore.pianoRollStore.getLocal(e2)
        const notePoint = NotePoint.add(
          pianoRollStore.transform.getNotePoint(local),
          offset,
        )

        const newPosition = quantize
          ? {
              tick: quantizer.round(notePoint.tick),
              noteNumber: notePoint.noteNumber,
            }
          : notePoint

        if (
          !pianoRollStore.validateDraggablePosition(
            draggable,
            newPosition,
            minLength,
          )
        ) {
          return
        }

        if (!isChanged) {
          isChanged = true
          pushHistory()
        }

        pianoRollStore.updateDraggable(draggable, () => newPosition)
      },
    })
  }
