import { AnyEvent } from "midifile-ts"
import { DistributiveOmit } from "./types.js"

export type PlayerEventOf<T> = DistributiveOmit<T, "deltaTime"> & {
  tick: number
  trackIndex: number
}

export type PlayerEvent = PlayerEventOf<AnyEvent>
