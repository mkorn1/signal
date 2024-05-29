import { IPoint } from "../geometry"

export const getClientPos = (e: MouseEvent): IPoint => ({
  x: e.clientX,
  y: e.clientY,
})
