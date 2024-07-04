import { Point } from "../geometry/Point"
import { TickTransform } from "./TickTransform"

export class TempoCoordTransform implements TickTransform {
  constructor(
    private readonly pixelsPerTick: number,
    // The height of the drawing area of the graph
    readonly height: number,
    readonly maxBPM = 320,
  ) {}

  getX(tick: number) {
    return tick * this.pixelsPerTick
  }

  getY(bpm: number) {
    return (1 - bpm / this.maxBPM) * this.height // 上下反転
  }

  getMaxY() {
    return this.height
  }

  getTick(pixels: number) {
    return pixels / this.pixelsPerTick
  }

  getBPM(pixels: number) {
    return (1 - pixels / this.height) * this.maxBPM
  }

  getDeltaBPM(pixels: number) {
    return (-pixels / this.height) * this.maxBPM
  }

  equals(t: TempoCoordTransform) {
    return (
      this.pixelsPerTick === t.pixelsPerTick &&
      this.height === t.height &&
      this.maxBPM === t.maxBPM
    )
  }

  fromPosition(position: Point) {
    return {
      tick: this.getTick(position.x),
      bpm: this.getBPM(position.y),
    }
  }
}
