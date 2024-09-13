import {
  cloneSelection,
  fixSelection,
  moveSelectionBy,
  resizeSelection,
  startSelection,
} from "../../../actions"
import { pushHistory } from "../../../actions/history"
import { Point } from "../../../entities/geometry/Point"
import { Rect } from "../../../entities/geometry/Rect"
import { NotePoint } from "../../../entities/transform/NotePoint"
import { observeDrag, observeDrag2 } from "../../../helpers/observeDrag"
import { PianoRollDraggable } from "../../../stores/PianoRollStore"
import RootStore from "../../../stores/RootStore"
import { MouseGesture } from "./NoteMouseHandler"

const MIN_LENGTH = 10

export const getSelectionActionForMouseDown =
  (rootStore: RootStore) =>
  (e: MouseEvent): MouseGesture | null => {
    if (e.relatedTarget) {
      return null
    }

    const { selectionBounds } = rootStore.pianoRollStore
    const local = rootStore.pianoRollStore.getLocal(e)

    if (e.button === 0) {
      if (selectionBounds !== null) {
        const type = positionType(selectionBounds, local)
        switch (type) {
          case "center":
            return moveSelectionAction(selectionBounds)
          case "right":
            return dragSelectionRightEdgeAction
          case "left":
            return dragSelectionLeftEdgeAction
          case "outside":
            return createSelectionAction
        }
      } else {
        return createSelectionAction
      }
    }

    return null
  }

export const getSelectionCursorForMouseMoven =
  (rootStore: RootStore) => (e: MouseEvent) => {
    const { selectionBounds } = rootStore.pianoRollStore
    const local = rootStore.pianoRollStore.getLocal(e)
    const type =
      selectionBounds === null
        ? "outside"
        : positionType(selectionBounds, local)
    switch (type) {
      case "center":
        return "move"
      case "left":
        return "w-resize"
      case "right":
        return "e-resize"
      case "outside":
        return "crosshair"
    }
  }

function positionType(selectionBounds: Rect, pos: Point) {
  const rect = selectionBounds
  const contains =
    rect.x <= pos.x &&
    rect.x + rect.width >= pos.x &&
    rect.y <= pos.y &&
    rect.y + rect.height >= pos.y
  if (!contains) {
    return "outside"
  }
  const localX = pos.x - rect.x
  const edgeSize = Math.min(rect.width / 3, 8)
  if (localX <= edgeSize) {
    return "left"
  }
  if (rect.width - localX <= edgeSize) {
    return "right"
  }
  return "center"
}

// 選択範囲外でクリックした場合は選択範囲をリセット
const createSelectionAction: MouseGesture = (rootStore) => (e) => {
  const {
    pianoRollStore: { transform, quantizer },
  } = rootStore

  const local = rootStore.pianoRollStore.getLocal(e)
  const start = transform.getNotePointFractional(local)
  const startPos = local
  startSelection(rootStore)(start)

  observeDrag2(e, {
    onMouseMove: (_e, delta) => {
      const offsetPos = Point.add(startPos, delta)
      const end = transform.getNotePointFractional(offsetPos)
      resizeSelection(rootStore)(
        { ...start, tick: quantizer.round(start.tick) },
        { ...end, tick: quantizer.round(end.tick) },
      )
    },

    onMouseUp: () => {
      fixSelection(rootStore)()
    },
  })
}

const moveSelectionAction =
  (selectionBounds: Rect): MouseGesture =>
  (rootStore) =>
  (e) => {
    const { transform, quantizer } = rootStore.pianoRollStore

    const isCopy = e.ctrlKey

    if (isCopy) {
      cloneSelection(rootStore)()
    }

    let isChanged = false
    let prevTick = 0
    let prevNoteNumber = 0

    observeDrag2(e, {
      onMouseMove: (_e, delta) => {
        const tick = quantizer.round(transform.getTick(delta.x))
        const noteNumber = Math.round(transform.getDeltaNoteNumber(delta.y))
        if (tick !== prevTick || noteNumber !== prevNoteNumber) {
          if (!isChanged) {
            isChanged = true
            pushHistory(rootStore)()
          }

          moveSelectionBy(rootStore)({
            tick: tick - prevTick,
            noteNumber: noteNumber - prevNoteNumber,
          })

          prevTick = tick
          prevNoteNumber = noteNumber
        }
      },
    })
  }

const dragSelectionEdgeAction =
  (edge: "left" | "right"): MouseGesture =>
  (rootStore) =>
  (e) => {
    const {
      pianoRollStore,
      pianoRollStore: { isQuantizeEnabled, quantizer },
      pushHistory,
    } = rootStore

    let isChanged = false

    const draggable: PianoRollDraggable = {
      type: "selection",
      position: edge,
    }

    const draggablePosition = pianoRollStore.getDraggablePosition(draggable)

    if (draggablePosition === null) {
      return
    }

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

const dragSelectionLeftEdgeAction = dragSelectionEdgeAction("left")
const dragSelectionRightEdgeAction = dragSelectionEdgeAction("right")
