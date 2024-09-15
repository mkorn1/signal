import { MaxNoteNumber } from "../../Constants"

export interface NotePoint {
  tick: number
  noteNumber: number
}

export namespace NotePoint {
  export const clampNoteNumber = (noteNumber: number) =>
    Math.min(MaxNoteNumber, Math.max(0, noteNumber))

  export const clamp = (point: NotePoint): NotePoint => ({
    tick: Math.max(0, point.tick),
    noteNumber: clampNoteNumber(point.noteNumber),
  })

  export function equals(a: NotePoint, b: NotePoint): boolean {
    return a.tick === b.tick && a.noteNumber === b.noteNumber
  }

  export function sub(a: NotePoint, b: NotePoint): NotePoint {
    return {
      tick: a.tick - b.tick,
      noteNumber: a.noteNumber - b.noteNumber,
    }
  }

  export function add(a: NotePoint, b: NotePoint): NotePoint {
    return {
      tick: a.tick + b.tick,
      noteNumber: a.noteNumber + b.noteNumber,
    }
  }
}
