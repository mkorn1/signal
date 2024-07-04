import { computed, makeObservable, observable } from "mobx"
import { BeatWithX } from "../entities/beat/BeatWithX"
import { TickTransform } from "../entities/transform/TickTransform"
import { filterEventsWithScroll } from "../helpers/filterEvents"
import Quantizer from "../quantizer"
import Song from "../song"
import { isTimeSignatureEvent } from "../track"

interface RulerProvider {
  rootStore: { song: Song }
  transform: TickTransform
  scrollLeft: number
  canvasWidth: number
  quantizer: Quantizer
}

export interface TimeSignature {
  id: number
  tick: number
  numerator: number
  denominator: number
  isSelected: boolean
}

export class RulerStore {
  selectedTimeSignatureEventIds: number[] = []

  constructor(readonly parent: RulerProvider) {
    makeObservable(this, {
      selectedTimeSignatureEventIds: observable.shallow,
      beats: computed,
      timeSignatures: computed,
      quantizer: computed,
    })
  }

  get beats(): BeatWithX[] {
    const { scrollLeft, transform, canvasWidth, rootStore } = this.parent

    return BeatWithX.createInRange(
      rootStore.song.measures,
      transform,
      rootStore.song.timebase,
      scrollLeft,
      canvasWidth,
    )
  }

  get timeSignatures(): TimeSignature[] {
    const { transform, scrollLeft, canvasWidth, rootStore } = this.parent
    const { selectedTimeSignatureEventIds } = this
    const track = rootStore.song.conductorTrack
    if (track === undefined) {
      return []
    }

    return filterEventsWithScroll(
      track.events,
      transform.getTick(scrollLeft),
      transform.getTick(canvasWidth),
    )
      .filter(isTimeSignatureEvent)
      .map((e) => ({
        ...e,
        isSelected: selectedTimeSignatureEventIds.includes(e.id),
      }))
  }

  get quantizer(): Quantizer {
    return this.parent.quantizer
  }

  getTick(offsetX: number) {
    const { transform, scrollLeft } = this.parent
    return transform.getTick(offsetX + scrollLeft)
  }

  getQuantizedTick(offsetX: number) {
    const { quantizer } = this.parent
    return quantizer.round(this.getTick(offsetX))
  }
}
