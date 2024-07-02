import { IPoint } from "../entities/geometry/Point"

export const getClientPos = (e: MouseEvent): IPoint => ({
  x: e.clientX,
  y: e.clientY,
})
