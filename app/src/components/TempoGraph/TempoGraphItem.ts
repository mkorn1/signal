import { IRect } from "../../entities/geometry/Rect"

export interface TempoGraphItem {
  id: number
  bounds: IRect
  microsecondsPerBeat: number
}
