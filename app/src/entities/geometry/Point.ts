export interface Point {
  x: number
  y: number
}

export function pointSub(v1: Point, v2: Point) {
  return {
    x: v1.x - v2.x,
    y: v1.y - v2.y,
  }
}

export function pointAdd(v1: Point, v2: Point) {
  return {
    x: v1.x + v2.x,
    y: v1.y + v2.y,
  }
}

export const zeroPoint = { x: 0, y: 0 }
