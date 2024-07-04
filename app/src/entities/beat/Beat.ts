import { Range } from "../geometry/Range"
import { MeasureList } from "../measure/MeasureList"

export interface Beat {
  measure: number
  beat: number
  tick: number
}

export namespace Beat {
  export const createInRange = (
    allMeasures: MeasureList,
    timebase: number,
    tickRange: Range,
  ): Beat[] => {
    const beats: Beat[] = []
    const measures = MeasureList.getMeasuresInRange(allMeasures, tickRange)

    measures.forEach((measure, i) => {
      const nextMeasure = measures[i + 1]

      const ticksPerBeat = (timebase * 4) / measure.denominator

      // 次の小節か曲の endTick まで拍を作る
      // Make a beat up to the next bar or song EndTick
      const lastTick = nextMeasure ? nextMeasure.startTick : tickRange[1]

      const startBeat = Math.max(
        0,
        Math.floor((tickRange[0] - measure.startTick) / ticksPerBeat),
      )
      const endBeat = (lastTick - measure.startTick) / ticksPerBeat

      for (let beat = startBeat; beat < endBeat; beat++) {
        const tick = measure.startTick + ticksPerBeat * beat
        beats.push({
          measure: measure.measure + Math.floor(beat / measure.numerator),
          beat: beat % measure.numerator,
          tick,
        })
      }
    })

    return beats
  }
}
