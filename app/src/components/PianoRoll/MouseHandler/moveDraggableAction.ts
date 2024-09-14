import { pick } from "lodash"
import { NotePoint } from "../../../entities/transform/NotePoint"
import { isNotNull } from "../../../helpers/array"
import { observeDrag } from "../../../helpers/observeDrag"
import { intersection } from "../../../helpers/set"
import { PianoRollDraggable } from "../../../stores/PianoRollStore"
import { MouseGesture } from "./NoteMouseHandler"
import { MIN_LENGTH } from "./SelectionMouseHandler"

export interface MoveDraggableCallback {
  onChange?: (e: MouseEvent, changes: Set<keyof NotePoint>) => void
  onMouseUp?: (e: MouseEvent) => void
  onClick?: (e: MouseEvent) => void
}

export const moveDraggableAction =
  (
    draggable: PianoRollDraggable,
    subDraggables: PianoRollDraggable[] = [],
    callback: MoveDraggableCallback = {},
  ): MouseGesture =>
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

    const subDraggablePositions = subDraggables.map((subDraggable) =>
      pianoRollStore.getDraggablePosition(subDraggable),
    )

    observeDrag({
      onMouseMove: (e2) => {
        const quantize = !e2.shiftKey && isQuantizeEnabled
        const minLength = quantize ? quantizer.unit : MIN_LENGTH
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

        const validProps = pianoRollStore.validateDraggablePosition(
          draggable,
          newPosition,
          minLength,
        )

        if (validProps.size === 0) {
          return
        }

        const delta = NotePoint.sub(newPosition, draggablePosition)

        const newSubDraggablePositions = subDraggables.map((_, i) => {
          const subDraggablePosition = subDraggablePositions[i]

          if (subDraggablePosition === null) {
            return null
          }

          return NotePoint.add(subDraggablePosition, delta)
        })

        const subValidProps = newSubDraggablePositions
          .map((subDraggablePosition, i) =>
            subDraggablePosition !== null
              ? pianoRollStore.validateDraggablePosition(
                  subDraggables[i],
                  subDraggablePosition,
                  minLength,
                )
              : null,
          )
          .filter(isNotNull)
          .reduce(intersection, validProps)

        if (subValidProps.size === 0) {
          return
        }

        if (!isChanged) {
          isChanged = true
          pushHistory()
        }

        const pickValidProps = (notePoint: NotePoint) =>
          pick(notePoint, Array.from(validProps.values()))

        pianoRollStore.updateDraggable(draggable, pickValidProps(newPosition))

        subDraggables.forEach((subDraggable, i) => {
          const subDraggablePosition = newSubDraggablePositions[i]

          if (subDraggablePosition === null || subDraggablePosition === null) {
            return
          }

          pianoRollStore.updateDraggable(
            subDraggable,
            pickValidProps(subDraggablePosition),
          )
        })

        callback?.onChange?.(e2, validProps)
      },
      onMouseUp: (e2) => {
        callback?.onMouseUp?.(e2)
      },
      onClick: (e2) => {
        callback?.onClick?.(e2)
      },
    })
  }
