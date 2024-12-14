import { cloneSelection, eventsInSelection } from "../../../actions"
import { Point } from "../../../entities/geometry/Point"
import { Rect } from "../../../entities/geometry/Rect"
import { Selection } from "../../../entities/selection/Selection"
import { observeDrag2 } from "../../../helpers/observeDrag"
import { useStores } from "../../../hooks/useStores"
import { moveDraggableAction } from "./moveDraggableAction"
import { MouseGesture } from "./NoteMouseHandler"

export const MIN_LENGTH = 10

export const useSelectionGesture = () => {
  const rootStore = useStores()

  return {
    onMouseDown(e: MouseEvent) {
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
    },
    getCursor(e: MouseEvent) {
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
    },
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
    pianoRollStore,
    pianoRollStore: { transform, quantizer, selectedTrack },
    player,
    controlStore,
  } = rootStore

  if (selectedTrack === undefined) {
    return
  }

  const local = rootStore.pianoRollStore.getLocal(e)
  const start = transform.getNotePointFractional(local)
  const startPos = local

  if (!player.isPlaying) {
    player.position = quantizer.round(start.tick)
  }

  controlStore.selectedEventIds = []
  pianoRollStore.selection = Selection.fromPoints(start, start)

  observeDrag2(e, {
    onMouseMove: (_e, delta) => {
      const offsetPos = Point.add(startPos, delta)
      const end = transform.getNotePointFractional(offsetPos)
      pianoRollStore.selection = Selection.fromPoints(
        { ...start, tick: quantizer.round(start.tick) },
        { ...end, tick: quantizer.round(end.tick) },
      )
    },

    onMouseUp: () => {
      const { selection } = pianoRollStore
      if (selection === null) {
        return
      }

      // 選択範囲を確定して選択範囲内のノートを選択状態にする
      // Confirm the selection and select the notes in the selection state
      pianoRollStore.selectedNoteIds = eventsInSelection(
        selectedTrack.events,
        selection,
      ).map((e) => e.id)
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
