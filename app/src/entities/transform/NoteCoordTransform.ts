import { Point } from "../../entities/geometry/Point"
import { NoteEvent } from "../../track"
import { NotePoint } from "./NotePoint"
import { TickTransform } from "./TickTransform"

export class NoteCoordTransform implements TickTransform {
  constructor(
    readonly pixelsPerTick: number,
    readonly pixelsPerKey: number,
    readonly maxNoteNumber: number,
  ) {}

  // pixels

  getX(tick: number) {
    return tick * this.pixelsPerTick
  }

  getY(noteNumber: number) {
    return (this.maxNoteNumber - noteNumber) * this.pixelsPerKey
  }

  getDeltaY(deltaNoteNumber: number) {
    return -deltaNoteNumber * this.pixelsPerKey
  }

  // ticks

  getTick(pixels: number) {
    return pixels / this.pixelsPerTick
  }

  getNoteNumber(pixels: number) {
    return Math.ceil(this.getNoteNumberFractional(pixels))
  }

  getNoteNumberFractional(pixels: number) {
    return this.maxNoteNumber - pixels / this.pixelsPerKey
  }

  getDeltaNoteNumber(deltaPixels: number) {
    return -deltaPixels / this.pixelsPerKey
  }

  get numberOfKeys() {
    return this.maxNoteNumber + 1
  }

  //

  getMaxY() {
    return (this.maxNoteNumber + 1) * this.pixelsPerKey
  }

  getRect(note: NoteEvent) {
    return {
      x: this.getX(note.tick),
      y: this.getY(note.noteNumber),
      width: this.getX(note.duration),
      height: this.pixelsPerKey,
    }
  }

  getDrumRect(note: NoteEvent) {
    return {
      x: this.getX(note.tick) - this.pixelsPerKey / 2,
      y: this.getY(note.noteNumber),
      width: this.pixelsPerKey,
      height: this.pixelsPerKey,
    }
  }

  getNotePoint(pos: Point): NotePoint {
    return {
      tick: this.getTick(pos.x),
      noteNumber: this.getNoteNumber(pos.y),
    }
  }

  getNotePointFractional(pos: Point): NotePoint {
    return {
      tick: this.getTick(pos.x),
      noteNumber: this.getNoteNumberFractional(pos.y),
    }
  }

  equals(t: NoteCoordTransform) {
    return (
      this.pixelsPerKey === t.pixelsPerKey &&
      this.pixelsPerTick === t.pixelsPerTick &&
      this.maxNoteNumber === t.maxNoteNumber
    )
  }
}
