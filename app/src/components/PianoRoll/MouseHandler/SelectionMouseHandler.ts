import {
  cloneSelection,
  fixSelection,
  resizeSelection,
  startSelection,
} from "../../../actions"
import { Point } from "../../../entities/geometry/Point"
import { Rect } from "../../../entities/geometry/Rect"
import { observeDrag2 } from "../../../helpers/observeDrag"
import RootStore from "../../../stores/RootStore"
import { moveDraggableAction } from "./moveDraggableAction"
import { MouseGesture } from "./NoteMouseHandler"

export const MIN_LENGTH = 10

export const getSelectionActionForMouseDown =
  (rootStore: RootStore) =>
  (e: MouseEvent): MouseGesture | null => {
    if (e.relatedTarget) {
      return null
    }

    const { selectionBounds, selectedNoteIds } = rootStore.pianoRollStore
    const local = rootStore.pianoRollStore.getLocal(e)

    if (e.button === 0) {
      if (selectionBounds !== null) {
        const type = positionType(selectionBounds, local)
        switch (type) {
          case "center":
            return moveSelectionAction(selectedNoteIds)
          case "right":
            return dragSelectionRightEdgeAction(selectedNoteIds)
          case "left":
            return dragSelectionLeftEdgeAction(selectedNoteIds)
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
  (selectedNoteIds: number[]): MouseGesture =>
  (rootStore) =>
  (e) => {
    const isCopy = e.metaKey

    if (isCopy) {
      cloneSelection(rootStore)()
    }

    return moveDraggableAction(
      { type: "selection", position: "center" },
      selectedNoteIds.map((noteId) => ({
        type: "note",
        position: "center",
        noteId,
      })),
    )(rootStore)(e)
  }

const dragSelectionLeftEdgeAction = (selectedNoteIds: number[]) =>
  moveDraggableAction(
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

const dragSelectionRightEdgeAction = (selectedNoteIds: number[]) =>
  moveDraggableAction(
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
