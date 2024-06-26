import { FC } from "react"
import { useLocalization } from "../localize/useLocalization"

export const scaleValues = [
  "major",
  "minor",
  "harmonicMinor",
  "melodicMinor",
  "pentatonicMajor",
  "pentatonicMinor",
  "blues",
  "wholeTone",
  "chromatic",
  "diminished",
  "augmented",
  "majorPentatonic",
  "minorPentatonic",
  "majorBlues",
] as const

export type Scale = (typeof scaleValues)[number]

export interface KeySignature {
  key: number // 0 is C, 1 is C#, 2 is D, etc.
  scale: Scale
}

// An array of intervals contained in the scale. Intervals are represented by relative distances from the key.
export const getScaleIntervals = (scale: Scale): number[] => {
  switch (scale) {
    case "major":
      return [0, 2, 4, 5, 7, 9, 11]
    case "minor":
      return [0, 2, 3, 5, 7, 8, 10]
    case "harmonicMinor":
      return [0, 2, 3, 5, 7, 8, 11]
    case "melodicMinor":
      return [0, 2, 3, 5, 7, 9, 11]
    case "pentatonicMajor":
      return [0, 2, 4, 7, 9]
    case "pentatonicMinor":
      return [0, 3, 5, 7, 10]
    case "blues":
      return [0, 3, 5, 6, 7, 10]
    case "wholeTone":
      return [0, 2, 4, 6, 8, 10]
    case "chromatic":
      return Array.from({ length: 12 }, (_, i) => i)
    case "diminished":
      return [0, 2, 3, 5, 6, 8, 9, 11]
    case "augmented":
      return [0, 3, 4, 7, 8, 11]
    case "majorPentatonic":
      return [0, 2, 4, 7, 9]
    case "minorPentatonic":
      return [0, 3, 5, 7, 10]
    case "majorBlues":
      return [0, 2, 4, 5, 7, 9, 11]
  }
}

export const ScaleName: FC<{ scale: Scale }> = ({ scale }) => {
  const localized = useLocalization()
  switch (scale) {
    case "major":
      return localized["scale-major"]
    case "minor":
      return localized["scale-minor"]
    case "harmonicMinor":
      return localized["scale-harmonic-minor"]
    case "melodicMinor":
      return localized["scale-melodic-minor"]
    case "pentatonicMajor":
      return localized["scale-pentatonic-major"]
    case "pentatonicMinor":
      return localized["scale-pentatonic-minor"]
    case "blues":
      return localized["scale-blues"]
    case "wholeTone":
      return localized["scale-whole-tone"]
    case "chromatic":
      return localized["scale-chromatic"]
    case "diminished":
      return localized["scale-diminished"]
    case "augmented":
      return localized["scale-augmented"]
    case "majorPentatonic":
      return localized["scale-major-pentatonic"]
    case "minorPentatonic":
      return localized["scale-minor-pentatonic"]
    case "majorBlues":
      return localized["scale-major-blues"]
  }
}
