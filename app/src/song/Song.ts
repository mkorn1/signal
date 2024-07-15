import { PlayerEvent } from "@signal-app/player"
import pullAt from "lodash/pullAt"
import {
  action,
  computed,
  makeObservable,
  observable,
  reaction,
  transaction,
} from "mobx"
import { createModelSchema, list, object, primitive } from "serializr"
import { Measure } from "../entities/measure/Measure"
import { isNotUndefined } from "../helpers/array"
import { collectAllEvents } from "../player/collectAllEvents"
import Track, { isTimeSignatureEvent } from "../track"

const END_MARGIN = 480 * 30
const DEFAULT_TIME_BASE = 480

export default class Song {
  tracks: Track[] = []
  filepath: string = ""
  timebase: number = DEFAULT_TIME_BASE
  name: string = ""
  fileHandle: FileSystemFileHandle | null = null
  cloudSongId: string | null = null
  cloudSongDataId: string | null = null
  isSaved = true

  constructor() {
    makeObservable(this, {
      addTrack: action,
      removeTrack: action,
      insertTrack: action,
      conductorTrack: computed,
      measures: computed,
      timeSignatures: computed,
      endOfSong: computed,
      allEvents: computed({ keepAlive: true }),
      tracks: observable.shallow,
      filepath: observable,
      timebase: observable,
      name: observable,
      isSaved: observable,
    })

    reaction(
      () => [
        this.tracks.map((t) => ({ channel: t.channel, events: t.events })),
        this.name,
      ],
      () => (this.isSaved = false),
    )
  }

  insertTrack(t: Track, index: number) {
    // 最初のトラックは Conductor Track なので channel を設定しない
    if (t.channel === undefined && this.tracks.length > 0) {
      t.channel = t.channel || this.tracks.length - 1
    }
    this.tracks.splice(index, 0, t)
  }

  addTrack(t: Track) {
    this.insertTrack(t, this.tracks.length)
  }

  removeTrack(id: number) {
    transaction(() => {
      pullAt(this.tracks, id)
    })
  }

  moveTrack(from: number, to: number) {
    const [track] = this.tracks.splice(from, 1)
    this.tracks.splice(to, 0, track)
  }

  get conductorTrack(): Track | undefined {
    return this.tracks.find((t) => t.isConductorTrack)
  }

  getTrack(id: number): Track | undefined {
    return this.tracks[id]
  }

  get measures(): Measure[] {
    const { timeSignatures, timebase } = this
    return Measure.fromTimeSignatures(timeSignatures, timebase)
  }

  get timeSignatures() {
    const { conductorTrack } = this
    if (conductorTrack === undefined) {
      return []
    }
    return conductorTrack.events
      .filter(isTimeSignatureEvent)
      .slice()
      .sort((a, b) => a.tick - b.tick)
  }

  get endOfSong(): number {
    const eos = Math.max(
      ...this.tracks.map((t) => t.endOfTrack).filter(isNotUndefined),
    )
    return (eos ?? 0) + END_MARGIN
  }

  get allEvents(): PlayerEvent[] {
    return collectAllEvents(this.tracks)
  }
}

createModelSchema(Song, {
  tracks: list(object(Track)),
  filepath: primitive(),
  timebase: primitive(),
})
