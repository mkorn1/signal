// An array of intervals contained in the scale. Intervals are represented by relative distances from the key.
export type Scale = number[]

export interface KeySignature {
  key: number // 0 is C, 1 is C#, 2 is D, etc.
  scale: Scale
}

export const majorScale = [0, 2, 4, 5, 7, 9, 11]
export const minorScale = [0, 2, 3, 5, 7, 8, 10]
export const harmonicMinorScale = [0, 2, 3, 5, 7, 8, 11]
export const melodicMinorScale = [0, 2, 3, 5, 7, 9, 11]
export const pentatonicMajorScale = [0, 2, 4, 7, 9]
export const pentatonicMinorScale = [0, 3, 5, 7, 10]
export const bluesScale = [0, 3, 5, 6, 7, 10]
export const wholeToneScale = [0, 2, 4, 6, 8, 10]
export const chromaticScale = Array.from({ length: 12 }, (_, i) => i)
export const diminishedScale = [0, 1, 3, 4, 6, 7, 9, 10]
export const augmentedScale = [0, 3, 4, 7, 8, 11]
export const majorPentatonicScale = [0, 2, 4, 7, 9]
export const minorPentatonicScale = [0, 3, 5, 7, 10]
export const majorBluesScale = [0, 2, 3, 4, 7, 9]
