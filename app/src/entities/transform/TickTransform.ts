export interface TickTransform {
  getX(tick: number): number
  getTick(x: number): number
}
