import { Point } from "../geometry/Point"
import { TickTransform } from "./TickTransform"

export class ArrangeCoordTransform implements TickTransform {
  constructor(
    readonly pixelsPerTick: number,
    readonly pixelsPerTrack: number,
  ) {}

  getX(tick: number): number {
    return tick * this.pixelsPerTick
  }

  getY(trackIndex: number): number {
    return trackIndex * this.pixelsPerTrack
  }

  getTick(x: number): number {
    return x / this.pixelsPerTick
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
