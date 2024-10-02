import { MaxNoteNumber } from "../../Constants"

export type NoteNumber = number

export namespace NoteNumber {
  export const clamp = (noteNumber: number) =>
    Math.min(MaxNoteNumber, Math.max(0, noteNumber))
}
