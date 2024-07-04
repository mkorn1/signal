import { Point } from "../geometry/Point"
import { TickTransform } from "./TickTransform"

export class ArrangeCoordTransform implements TickTransform {
  constructor(
    private readonly transform: TickTransform,
    private readonly pixelsPerTrack: number,
  ) {}

  getX(tick: number): number {
    return this.transform.getX(tick)
  }

  getY(trackIndex: number): number {
    return trackIndex * this.pixelsPerTrack
  }

  getTick(x: number): number {
    return this.transform.getTick(x)
  }

  getTrackIndex(y: number): number {
    return y / this.pixelsPerTrack
  }

  getArrangePoint(point: Point) {
    return {
      tick: this.getTick(point.x),
      trackIndex: this.getTrackIndex(point.y),
    }
  }
}
