import { cloneSelection, eventsInSelection } from "../../../actions"
import { Point } from "../../../entities/geometry/Point"
import { Rect } from "../../../entities/geometry/Rect"
import { Selection } from "../../../entities/selection/Selection"
import { observeDrag2 } from "../../../helpers/observeDrag"
import { useStores } from "../../../hooks/useStores"
import { useMoveDraggableGesture } from "./moveDraggableAction"

export const MIN_LENGTH = 10

export const useSelectionGesture = () => {
  const {
    pianoRollStore,
    pianoRollStore: { selectionBounds, selectedNoteIds },
  } = useStores()
  const moveSelectionAction = useMoveSelectionGesture()
  const dragSelectionLeftEdgeAction = useDragSelectionLeftEdgeGesture()
  const dragSelectionRightEdgeAction = useDragSelectionRightEdgeGesture()
  const createSelectionAction = useCreateSelectionGesture()

  return {
    onMouseDown(e: MouseEvent) {
      if (e.relatedTarget) {
        return null
      }

      const local = pianoRollStore.getLocal(e)

      if (e.button === 0) {
        if (selectionBounds !== null) {
          const type = positionType(selectionBounds, local)
          switch (type) {
            case "center":
              return moveSelectionAction.onMouseDown(e, selectedNoteIds)
            case "right":
              return dragSelectionRightEdgeAction.onMouseDown(
                e,
                selectedNoteIds,
              )
            case "left":
              return dragSelectionLeftEdgeAction.onMouseDown(e, selectedNoteIds)
            case "outside":
              return createSelectionAction.onMouseDown(e)
          }
        } else {
          return createSelectionAction.onMouseDown(e)
        }
      }

      return null
    },
    getCursor(e: MouseEvent) {
      const local = pianoRollStore.getLocal(e)
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
const useCreateSelectionGesture = () => {
  const {
    pianoRollStore,
    pianoRollStore: { transform, quantizer, selectedTrack },
    player,
    controlStore,
  } = useStores()

  return {
    onMouseDown(e: MouseEvent) {
      if (selectedTrack === undefined) {
        return
      }

      const local = pianoRollStore.getLocal(e)
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
    },
  }
}

const useMoveSelectionGesture = () => {
  const rootStore = useStores()
  const moveDraggableAction = useMoveDraggableGesture()

  return {
    onMouseDown(e: MouseEvent, selectedNoteIds: number[]) {
      const isCopy = e.metaKey

      if (isCopy) {
        cloneSelection(rootStore)()
      }

      return moveDraggableAction.onMouseDown(
        e,
        { type: "selection", position: "center" },
        selectedNoteIds.map((noteId) => ({
          type: "note",
          position: "center",
          noteId,
        })),
      )
    },
  }
}

const useDragSelectionLeftEdgeGesture = () => {
  const moveDraggableAction = useMoveDraggableGesture()

  return {
    onMouseDown(e: MouseEvent, selectedNoteIds: number[]) {
      moveDraggableAction.onMouseDown(
        e,
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
    },
  }
}

const useDragSelectionRightEdgeGesture = () => {
  const moveDraggableAction = useMoveDraggableGesture()

  return {
    onMouseDown(e: MouseEvent, selectedNoteIds: number[]) {
      moveDraggableAction.onMouseDown(
        e,
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
    },
  }
}
