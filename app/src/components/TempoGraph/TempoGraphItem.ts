import { IRect } from "../../geometry"

export interface TempoGraphItem {
  id: number
  bounds: IRect
  microsecondsPerBeat: number
}
