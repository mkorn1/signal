import { MeasureList } from "../measure/MeasureList"
import { Beat } from "./Beat"

export type BeatWithX = Beat & {
  x: number
}

export namespace BeatWithX {
  export const createInRange = (
    allMeasures: MeasureList,
    pixelsPerTick: number,
    timebase: number,
    startTick: number,
    width: number,
  ): BeatWithX[] => {
    const endTick = startTick + width / pixelsPerTick
    return Beat.createInRange(allMeasures, timebase, startTick, endTick).map(
      (b) => ({
        ...b,
        x: Math.round(b.tick * pixelsPerTick),
      }),
    )
  }
}
