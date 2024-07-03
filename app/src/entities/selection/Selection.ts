import cloneDeep from "lodash/cloneDeep"
import { MaxNoteNumber } from "../../Constants"
import { Range } from "../geometry/Range"
import { Rect } from "../geometry/Rect"
import { NoteCoordTransform } from "../transform/NoteCoordTransform"
import { NotePoint } from "../transform/NotePoint"

export interface Selection {
  from: NotePoint
  to: NotePoint
}

export namespace Selection {
  export const getBounds = (
    selection: Selection,
    transform: NoteCoordTransform,
  ): Rect => {
    const left = transform.getX(selection.from.tick)
    const right = transform.getX(selection.to.tick)
    const top = transform.getY(selection.from.noteNumber)
    const bottom = transform.getY(selection.to.noteNumber)
    return {
      x: left,
      y: top,
      width: right - left,
      height: bottom - top,
    }
  }

  export const clone = (selection: Selection): Selection => cloneDeep(selection)

  export const moved = (
    selection: Selection,
    dt: number,
    dn: number,
  ): Selection => {
    const s = clone(selection)

    s.from.tick += dt
    s.to.tick += dt
    s.from.noteNumber += dn
    s.to.noteNumber += dn

    return s
  }

  // to Make the lower right
  export const regularized = (
    fromTick: number,
    fromNoteNumber: number,
    toTick: number,
    toNoteNumber: number,
  ): Selection => ({
    from: {
      tick: Math.max(0, Math.min(fromTick, toTick)),
      noteNumber: Math.min(
        MaxNoteNumber,
        Math.max(fromNoteNumber, toNoteNumber),
      ),
    },
    to: {
      tick: Math.max(fromTick, toTick),
      noteNumber: Math.min(
        MaxNoteNumber,
        Math.min(fromNoteNumber, toNoteNumber),
      ),
    },
  })

  export const clamp = (selection: Selection): Selection => ({
    from: NotePoint.clamp(selection.from),
    to: NotePoint.clamp(selection.to),
  })

  export function fromPoints(start: NotePoint, end: NotePoint) {
    let selection = Selection.regularized(
      start.tick,
      start.noteNumber,
      end.tick,
      end.noteNumber,
    )

    // integer containing the original coordinates.
    selection.from.noteNumber = Math.ceil(selection.from.noteNumber)
    selection.to.noteNumber = Math.floor(selection.to.noteNumber)

    ++selection.to.noteNumber
    selection = Selection.clamp(selection)
    --selection.to.noteNumber

    return selection
  }

  // Fix the right end and change the length
  export function resizeLeft(
    selection: Selection,
    tick: number,
    minLength: number,
  ) {
    const range = Range.create(selection.from.tick, selection.to.tick)
    const [newFromTick, newToTick] = Range.resizeStart(range, tick, minLength)
    return Selection.regularized(
      newFromTick,
      selection.from.noteNumber,
      newToTick,
      selection.to.noteNumber,
    )
  }

  export function equals(a: Selection, b: Selection) {
    return NotePoint.equals(a.from, b.from) && NotePoint.equals(a.to, b.to)
  }
}
