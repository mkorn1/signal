import Track, { isTimeSignatureEvent } from "../track"
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
