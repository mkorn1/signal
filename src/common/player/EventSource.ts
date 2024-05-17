import { SongProvider } from "../song/SongProvider"
import { IEventSource } from "./Player"
import { PlayerEvent } from "./PlayerEvent"

export class EventSource implements IEventSource {
  constructor(private readonly songStore: SongProvider) {}

  get timebase(): number {
    return this.songStore.song.timebase
  }

  get endOfSong(): number {
    return this.songStore.song.endOfSong
  }

  get allEvents(): PlayerEvent[] {
    return this.songStore.song.allEvents
  }

  get measures() {
    return this.songStore.song.measures
  }

  get tracks() {
    return this.songStore.song.tracks
  }
}
