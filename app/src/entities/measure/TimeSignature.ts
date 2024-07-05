import { getLast, isTickBefore } from "../../track/selector"

export interface TimeSignature {
  tick: number
  numerator: number
  denominator: number
}

export namespace TimeSignature {
  const defaultValue: TimeSignature = {
    numerator: 4,
    denominator: 4,
    tick: 0,
  }

  function getAt(
    events: TimeSignature[],
    tick: number,
  ): TimeSignature | undefined {
    return getLast(events.filter(isTickBefore(tick)))
  }

  function getMeasureStart(
    events: TimeSignature[],
    tick: number,
    timebase: number,
  ) {
    const e = getAt(events, tick) ?? defaultValue
    const ticksPerMeasure = ((timebase * 4) / e.denominator) * e.numerator
    const measures = (tick - e.tick) / ticksPerMeasure
    const fixedMeasures = Math.floor(measures)
    const beginMeasureTick = e.tick + ticksPerMeasure * fixedMeasures
    const ticksPerBeat = (timebase * 4) / e.denominator
    return {
      tick: beginMeasureTick,
      duration: ticksPerMeasure,
      ticksPerBeat,
    }
  }

  /**
   * Returns the tick one measure before the specified tick
   *
   * If the tick is already at the beginning of a measure, it returns the tick of the previous measure
   *
   * To prevent the inability to rewind during playback,
   * if the position has not advanced more than 1 beat from the beginning of the measure,
   * rewind further to the previous measure
   */
  export function getPreviousMeasureTick(
    events: TimeSignature[],
    position: number,
    timebase: number,
  ): number {
    const measureStart = getMeasureStart(events, position, timebase)

    if (position > measureStart.tick + measureStart.ticksPerBeat) {
      return measureStart.tick
    }

    // previous measure
    return getMeasureStart(events, measureStart.tick - 1, timebase).tick
  }

  export function getNextMeasureTick(
    events: TimeSignature[],
    position: number,
    timebase: number,
  ): number {
    const measureStart = getMeasureStart(events, position, timebase)
    return measureStart.tick + measureStart.duration
  }
}
