import { IRect } from "../../../main/common/geometry"

export interface TempoGraphItem {
  id: number
  bounds: IRect
  microsecondsPerBeat: number
}
