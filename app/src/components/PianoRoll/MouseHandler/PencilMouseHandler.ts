import {
  addNoteToSelection,
  createNote,
  fixSelection,
  removeEvent,
  removeNoteFromSelection,
  resizeSelection,
  selectNote,
  startNote,
  startSelection,
  stopNote,
} from "../../../actions"
import { Point } from "../../../entities/geometry/Point"
import { observeDrag2 } from "../../../helpers/observeDrag"
import { useStores } from "../../../hooks/useStores"
import { PianoNoteItem } from "../../../stores/PianoRollStore"
import { isNoteEvent } from "../../../track"
import { moveDraggableAction } from "./moveDraggableAction"
import { MouseGesture } from "./NoteMouseHandler"

export const usePencilGesture = () => {
  const rootStore = useStores()

  return {
    onMouseDown(e: MouseEvent) {
      const local = rootStore.pianoRollStore.getLocal(e)
      const items = rootStore.pianoRollStore.getNotes(local)
      const isDrum =
        rootStore.pianoRollStore.selectedTrack?.isRhythmTrack ?? false

      switch (e.button) {
        case 0: {
          if (items.length > 0) {
            if (e.detail % 2 === 0) {
              return removeNoteAction
            }

            const item = items[0]

            if (e.shiftKey) {
              if (item.isSelected) {
                removeNoteFromSelection(rootStore)(item.id)
              } else {
                addNoteToSelection(rootStore)(item.id)
              }
            } else {
              if (!item.isSelected) {
                selectNote(rootStore)(item.id)
              }
              const position = getPositionType(local, item, isDrum)
              switch (position) {
                case "center":
                  return dragNoteCenterAction(item.id)
                case "left":
                  return dragNoteLeftAction(item.id)
                case "right":
                  return dragNoteRightAction(item.id)
              }
            }
          } else {
            if (e.shiftKey || e.metaKey) {
              return selectNoteAction
            }
          }

          return createNoteAction
        }
        case 2:
          return removeNoteAction
        default:
          return null
      }
    },
    getCursor(e: MouseEvent): string {
      const local = rootStore.pianoRollStore.getLocal(e)
      const items = rootStore.pianoRollStore.getNotes(local)
      const isDrum =
        rootStore.pianoRollStore.selectedTrack?.isRhythmTrack ?? false
      if (items.length > 0) {
        const position = getPositionType(local, items[0], isDrum)
        return mousePositionToCursor(position)
      }

      return "auto"
    },
  }
}

type MousePositionType = "left" | "center" | "right"

const mousePositionToCursor = (position: MousePositionType) => {
  switch (position) {
    case "center":
      return "move"
    case "left":
      return "w-resize"
    case "right":
      return "e-resize"
  }
}

const getPositionType = (
  local: Point,
  item: PianoNoteItem,
  isDrum: boolean,
): MousePositionType => {
  if (item === null) {
    console.warn("no item")
    return "center"
  }
  const localX = local.x - item.x

  if (isDrum) {
    return "center"
  }
  const edgeSize = Math.min(item.width / 3, 8)
  if (localX <= edgeSize) {
    return "left"
  }
  if (item.width - localX <= edgeSize) {
    return "right"
  }
  return "center"
}

const dragNoteEdgeAction =
  (edge: "left" | "right" | "center") =>
  (noteId: number): MouseGesture =>
  (rootStore) =>
  (e) => {
    const {
      pianoRollStore,
      pianoRollStore: { selectedTrack, selectedNoteIds },
    } = rootStore

    if (selectedTrack === undefined || selectedTrack.channel === undefined) {
      return
    }

    const note = selectedTrack.getEventById(noteId)
    if (note == undefined || !isNoteEvent(note)) {
      return
    }

    const { channel } = selectedTrack
    startNote(rootStore)({ ...note, channel })
    let playingNoteNumber = note.noteNumber

    moveDraggableAction(
      { type: "note", position: edge, noteId },
      selectedNoteIds
        .filter((id) => id !== noteId)
        .map((noteId) => ({
          type: "note",
          position: edge,
          noteId,
        })),
      {
        onChange(_e, { oldPosition, newPosition }) {
          const newNote = selectedTrack.getEventById(noteId)
          if (newNote == undefined || !isNoteEvent(newNote)) {
            return
          }
          // save last note duration
          if (oldPosition.tick !== newPosition.tick) {
            pianoRollStore.lastNoteDuration = newNote.duration
          }
          if (
            oldPosition.noteNumber !== newPosition.noteNumber &&
            newNote.noteNumber !== playingNoteNumber
          ) {
            stopNote(rootStore)({ noteNumber: playingNoteNumber, channel })
            startNote(rootStore)({
              noteNumber: newNote.noteNumber,
              channel,
              velocity: newNote.velocity,
            })
            playingNoteNumber = newNote.noteNumber
          }
        },
        onMouseUp() {
          stopNote(rootStore)({ noteNumber: playingNoteNumber, channel })
        },
        onClick(e) {
          if (!e.shiftKey) {
            selectNote(rootStore)(noteId)
          }
        },
      },
    )(rootStore)(e)
  }

const dragNoteLeftAction = dragNoteEdgeAction("left")
const dragNoteRightAction = dragNoteEdgeAction("right")
const dragNoteCenterAction = dragNoteEdgeAction("center")

const createNoteAction: MouseGesture = (rootStore) => (e) => {
  const { transform } = rootStore.pianoRollStore
  const local = rootStore.pianoRollStore.getLocal(e)

  if (e.shiftKey) {
    return
  }

  const { tick, noteNumber } = transform.getNotePoint(local)
  const note = createNote(rootStore)(tick, noteNumber)

  if (note === undefined) {
    return
  }

  selectNote(rootStore)(note.id)
  dragNoteCenterAction(note.id)(rootStore)(e)
}

const removeNoteAction: MouseGesture = (rootStore) => (e) => {
  const startPos = rootStore.pianoRollStore.getLocal(e)
  const items = rootStore.pianoRollStore.getNotes(startPos)
  if (items.length > 0) {
    removeEvent(rootStore)(items[0].id)
  }

  observeDrag2(e, {
    onMouseMove: (_e, delta) => {
      const local = Point.add(startPos, delta)
      const items = rootStore.pianoRollStore.getNotes(local)
      if (items.length > 0) {
        removeEvent(rootStore)(items[0].id)
      }
    },
  })
}

const selectNoteAction: MouseGesture = (rootStore) => (e) => {
  const {
    pianoRollStore: { transform },
  } = rootStore

  const local = rootStore.pianoRollStore.getLocal(e)
  const start = transform.getNotePoint(local)
  const startPos = local
  startSelection(rootStore)(start, true)

  observeDrag2(e, {
    onMouseMove: (_e, delta) => {
      const offsetPos = Point.add(startPos, delta)
      const end = transform.getNotePoint(offsetPos)
      resizeSelection(rootStore)(start, end)
    },

    onMouseUp: () => {
      fixSelection(rootStore)(true)
    },
  })
}
