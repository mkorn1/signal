import { Scale, getScaleIntegerNotation } from "./Scale"

export interface KeySignature {
  key: number // 0 is C, 1 is C#, 2 is D, etc.
  scale: Scale
} // the function that transpose the scale to the key

export const getScaleInterval = (keySignature: KeySignature): number[] => {
  const scaleIntervals = getScaleIntegerNotation(keySignature.scale)
  return scaleIntervals.map((i) => (i + keySignature.key) % 12)
}
