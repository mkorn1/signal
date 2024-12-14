import {
  addNoteToSelection,
  eventsInSelection,
  removeNoteFromSelection,
  selectNote,
  startNote,
  stopNote,
} from "../../../actions"
import { Point } from "../../../entities/geometry/Point"
import { Selection } from "../../../entities/selection/Selection"
import { NoteNumber } from "../../../entities/unit/NoteNumber"
import { observeDrag2 } from "../../../helpers/observeDrag"
import { useStores } from "../../../hooks/useStores"
import { PianoNoteItem } from "../../../stores/PianoRollStore"
import { isNoteEvent, NoteEvent } from "../../../track"
import { moveDraggableAction } from "./moveDraggableAction"

const createGesture = <Params extends any[]>(
  fn: (e: MouseEvent, ...params: Params) => void,
) => ({
  onMouseDown: fn,
})

export const usePencilGesture = () => {
  const rootStore = useStores()
  const removeNoteGesture = useRemoveNoteGesture()
  const createNoteGesture = useCreateNoteGesture()
  const selectNoteGesture = useSelectNoteGesture()
  const dragNoteCenterGesture = useDragNoteCenterGesture()
  const dragNoteLeftGesture = useDragNoteLeftGesture()
  const dragNoteRightGesture = useDragNoteRightGesture()
  const removeNoteFromSelectionGesture = createGesture((_e, noteId: number) =>
    removeNoteFromSelection(rootStore)(noteId),
  )
  const addNoteToSelectionGesture = createGesture((_e, noteId: number) =>
    addNoteToSelection(rootStore)(noteId),
  )

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
              return removeNoteGesture.onMouseDown(e)
            }

            const item = items[0]

            if (e.shiftKey) {
              if (item.isSelected) {
                removeNoteFromSelectionGesture.onMouseDown(e, item.id)
              } else {
                addNoteToSelectionGesture.onMouseDown(e, item.id)
              }
            } else {
              const position = getPositionType(local, item, isDrum)
              switch (position) {
                case "center":
                  return dragNoteCenterGesture.onMouseDown(e, item.id)
                case "left":
                  return dragNoteLeftGesture.onMouseDown(e, item.id)
                case "right":
                  return dragNoteRightGesture.onMouseDown(e, item.id)
              }
            }
          } else {
            if (e.shiftKey || e.metaKey) {
              return selectNoteGesture.onMouseDown(e)
            } else {
              return createNoteGesture.onMouseDown(e)
            }
          }
          break
        }
        case 2:
          return removeNoteGesture.onMouseDown(e)
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

const useDragNoteEdgeGesture = (edge: "left" | "right" | "center") => () => {
  const rootStore = useStores()

  return {
    onMouseDown(e: MouseEvent, noteId: number) {
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

      const isSelected = selectedNoteIds.includes(noteId)

      if (!isSelected) {
        selectNote(rootStore)(noteId)
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
    },
  }
}

const useDragNoteLeftGesture = useDragNoteEdgeGesture("left")
const useDragNoteRightGesture = useDragNoteEdgeGesture("right")
const useDragNoteCenterGesture = useDragNoteEdgeGesture("center")

const useCreateNoteGesture = () => {
  const rootStore = useStores()
  const {
    song: { timebase },
    pianoRollStore,
    pianoRollStore: { transform, selectedTrack, quantizer, newNoteVelocity },
    pushHistory,
  } = rootStore
  const dragNoteCenterAction = useDragNoteCenterGesture()

  return {
    onMouseDown(e: MouseEvent) {
      if (e.shiftKey) {
        return
      }

      const local = pianoRollStore.getLocal(e)
      const { tick, noteNumber } = transform.getNotePoint(local)

      if (
        selectedTrack === undefined ||
        selectedTrack.channel == undefined ||
        !NoteNumber.isValid(noteNumber)
      ) {
        return
      }

      pushHistory()

      const quantizedTick = selectedTrack.isRhythmTrack
        ? quantizer.round(tick)
        : quantizer.floor(tick)

      const duration = selectedTrack.isRhythmTrack
        ? timebase / 8 // 32th note in the rhythm track
        : (pianoRollStore.lastNoteDuration ?? quantizer.unit)

      const note = selectedTrack.addEvent({
        type: "channel",
        subtype: "note",
        noteNumber: noteNumber,
        tick: quantizedTick,
        velocity: newNoteVelocity,
        duration,
      } as NoteEvent)

      if (note === undefined) {
        return
      }

      selectNote(rootStore)(note.id)
      dragNoteCenterAction.onMouseDown(e, note.id)
    },
  }
}

const useRemoveNoteGesture = () => {
  const {
    pianoRollStore,
    pianoRollStore: { selectedTrack },
    pushHistory,
  } = useStores()

  const removeEvent = (eventId: number) => {
    if (selectedTrack === undefined) {
      return
    }
    pushHistory()
    selectedTrack.removeEvent(eventId)
    pianoRollStore.selectedNoteIds = pianoRollStore.selectedNoteIds.filter(
      (id) => id !== eventId,
    )
  }

  return {
    onMouseDown(e: MouseEvent) {
      const startPos = pianoRollStore.getLocal(e)
      const items = pianoRollStore.getNotes(startPos)
      if (items.length > 0) {
        removeEvent(items[0].id)
      }

      observeDrag2(e, {
        onMouseMove: (_e, delta) => {
          const local = Point.add(startPos, delta)
          const items = pianoRollStore.getNotes(local)
          if (items.length > 0) {
            removeEvent(items[0].id)
          }
        },
      })
    },
  }
}

const useSelectNoteGesture = () => {
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
      const start = transform.getNotePoint(local)
      const startPos = local

      if (!player.isPlaying) {
        player.position = quantizer.round(start.tick)
      }

      controlStore.selectedEventIds = []
      pianoRollStore.selection = Selection.fromPoints(start, start)

      observeDrag2(e, {
        onMouseMove: (_e, delta) => {
          const offsetPos = Point.add(startPos, delta)
          const end = transform.getNotePoint(offsetPos)
          pianoRollStore.selection = Selection.fromPoints(start, end)
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

          pianoRollStore.selection = null
        },
      })
    },
  }
}
