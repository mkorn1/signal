import { Span } from "./Span"

export type Range = [number, number]

export namespace Range {
  export function create(start: number, end: number): Range {
    return [start, end]
  }

  // move the start of the range without changing the end
  export function resizeStart(
    range: Range,
    start: number,
    minLength: number,
    minStart: number = 0,
  ): Range {
    const [, end] = range
    const newStart = Math.max(minStart, Math.min(start, end - minLength))
    return [newStart, end]
  }

  export function resizeEnd(
    range: Range,
    end: number,
    minLength: number,
    maxEnd: number = Infinity,
  ): Range {
    const [start] = range
    const newEnd = Math.min(maxEnd, Math.max(end, start + minLength))
    return [start, newEnd]
  }

  export function toSpan([start, end]: Range): Span {
    return {
      start,
      length: end - start,
    }
  }

  export function equals(a: Range, b: Range) {
    return a[0] === b[0] && a[1] === b[1]
  }
}
