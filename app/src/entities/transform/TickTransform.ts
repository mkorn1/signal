export interface TickTransform {
  pixelsPerTick: number
  getX(tick: number): number
  getTick(x: number): number
}
