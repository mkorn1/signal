import Track, { isTimeSignatureEvent } from "../../track"
import { Range } from "../geometry/Range"
import { Measure } from "./Measure"

export type MeasureList = Measure[]

export namespace MeasureList {
  export function getMeasureAt(measures: Measure[], tick: number): Measure {
    let lastMeasure: Measure = {
      startTick: 0,
      measure: 0,
      denominator: 4,
      numerator: 4,
    }
    for (const m of measures) {
      if (m.startTick > tick) {
        break
      }
      lastMeasure = m
    }
    return lastMeasure
  }

  export function fromConductorTrack(
    conductorTrack: Track,
    timebase: number,
  ): Measure[] {
    const events = conductorTrack.events
      .filter(isTimeSignatureEvent)
      .slice()
      .sort((a, b) => a.tick - b.tick)

    if (events.length === 0) {
      return [
        {
          startTick: 0,
          measure: 0,
          denominator: 4,
          numerator: 4,
        },
      ]
    } else {
      let lastMeasure = 0
      return events.map((e, i) => {
        let measure = 0
        if (i > 0) {
          const lastEvent = events[i - 1]
          const ticksPerBeat = (timebase * 4) / lastEvent.denominator
          const measureDelta = Math.floor(
            (e.tick - lastEvent.tick) / ticksPerBeat / lastEvent.numerator,
          )
          measure = lastMeasure + measureDelta
          lastMeasure = measure
        }
        return {
          startTick: e.tick,
          measure,
          numerator: e.numerator,
          denominator: e.denominator,
        }
      })
    }
  }

  export const getMBTString = (
    measures: Measure[],
    tick: number,
    ticksPerBeat: number,
    formatter = defaultMBTFormatter,
  ): string => formatter(getMBT(measures, tick, ticksPerBeat))

  // Find the measure in the range. The first element also includes those before startTick
  export const getMeasuresInRange = (measures: Measure[], tickRange: Range) => {
    let i = 0
    const result: Measure[] = []

    for (const measure of measures) {
      const nextMeasure = measures[i + 1]
      i++

      // Find the first measure
      if (result.length === 0) {
        if (
          nextMeasure !== undefined &&
          nextMeasure.startTick <= tickRange[0]
        ) {
          // Skip if the next Measure can be the first
          continue
        }
        if (measure.startTick > tickRange[0]) {
          console.warn("There is no initial time signature. Use 4/4 by default")
          result.push({
            startTick: 0,
            measure: 0,
            numerator: 4,
            denominator: 4,
          })
        } else {
          result.push(measure)
        }
      }

      // Find the remaining measures. Check if there is another first measure again to handle the case where there is no first measure correctly.
      if (result.length !== 0) {
        if (measure.startTick <= tickRange[1]) {
          result.push(measure)
        } else {
          break
        }
      }
    }

    return result
  }
}

interface Beat {
  measure: number
  beat: number
  tick: number
}

const getMBT = (
  measures: Measure[],
  tick: number,
  ticksPerBeat: number,
): Beat => {
  return Measure.calculateMBT(
    MeasureList.getMeasureAt(measures, tick),
    tick,
    ticksPerBeat,
  )
}

const pad = (v: number, digit: number) => {
  const str = v.toFixed(0)
  return ("0".repeat(digit) + str).slice(-Math.max(digit, str.length))
}

function defaultMBTFormatter(mbt: Beat): string {
  return `${pad(mbt.measure + 1, 4)}:${pad(mbt.beat + 1, 2)}:${pad(
    mbt.tick,
    3,
  )}`
}
