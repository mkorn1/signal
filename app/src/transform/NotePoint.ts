import { MaxNoteNumber } from "../Constants"

export interface NotePoint {
  tick: number
  noteNumber: number
}

export const clampNoteNumber = (noteNumber: number) =>
  Math.min(MaxNoteNumber, Math.max(0, noteNumber))

export const clampNotePoint = (point: NotePoint): NotePoint => ({
  tick: Math.max(0, point.tick),
  noteNumber: clampNoteNumber(point.noteNumber),
})
