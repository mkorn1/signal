import {
  cloneSelection,
  fixSelection,
  moveSelection,
  resizeSelection,
  resizeSelectionLeft,
  resizeSelectionRight,
  startSelection,
} from "../../../actions"
import { pushHistory } from "../../../actions/history"
import { Point } from "../../../entities/geometry/Point"
import { Rect } from "../../../entities/geometry/Rect"
import { observeDrag, observeDrag2 } from "../../../helpers/observeDrag"
import RootStore from "../../../stores/RootStore"
import { MouseGesture } from "./NoteMouseHandler"

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
    const { transform } = rootStore.pianoRollStore

    const isCopy = e.ctrlKey

    if (isCopy) {
      cloneSelection(rootStore)()
    }

    let isMoved = false

    observeDrag2(e, {
      onMouseMove: (_e, delta) => {
        const position = Point.add(selectionBounds, delta)
        const tick = transform.getTicks(position.x)
        const noteNumber = transform.getNoteNumberFractional(position.y)

        if ((tick !== 0 || noteNumber !== 0) && !isMoved) {
          isMoved = true
          pushHistory(rootStore)()
        }

        moveSelection(rootStore)({
          tick,
          noteNumber,
        })
      },
    })
  }

const dragSelectionLeftEdgeAction: MouseGesture = (rootStore) => (e) => {
  observeDrag({
    onMouseMove: (e2) => {
      const local = rootStore.pianoRollStore.getLocal(e2)
      const tick = rootStore.pianoRollStore.transform.getTicks(local.x)
      resizeSelectionLeft(rootStore)(tick)
    },
  })
}

const dragSelectionRightEdgeAction: MouseGesture = (rootStore) => (e) => {
  observeDrag({
    onMouseMove: (e2) => {
      const local = rootStore.pianoRollStore.getLocal(e2)
      const tick = rootStore.pianoRollStore.transform.getTicks(local.x)
      resizeSelectionRight(rootStore)(tick)
    },
  })
}
