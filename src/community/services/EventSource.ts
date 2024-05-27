import {
  IEventSource,
  PlayerEvent,
  PlayerEventOf,
  SendableEvent,
} from "@signal-app/player"
import { maxBy } from "lodash"
import uniq from "lodash/uniq"
import { AnyChannelEvent } from "midifile-ts"
import { isNotUndefined } from "../../main/helpers/array"
import { filterEventsWithRange } from "../../main/helpers/filterEvents"
import { Song, TrackEvent } from "../song/Song"
import {
  isControllerEvent,
  isControllerEventWithType,
  isPitchBendEvent,
  isProgramChangeEvent,
  isSetTempoEvent,
} from "../song/identify"

export class EventSource implements IEventSource {
  constructor(private readonly songProvider: { song: Song }) {}

  get timebase(): number {
    return this.songProvider.song.timebase
  }

  get endOfSong(): number {
    return this.songProvider.song.endOfSong
  }

  getEvents(startTick: number, endTick: number): PlayerEvent[] {
    return this.songProvider.song.tracks.flatMap((track, trackId) =>
      filterEventsWithRange(track.events, startTick, endTick).map((event) => ({
        ...event,
        trackId,
      })),
    )
  }

  getCurrentStateEvents(tick: number): SendableEvent[] {
    return this.songProvider.song.tracks.flatMap((t, trackId) => {
      const statusEvents = getStatusEvents(t.events, tick)
      return statusEvents.map(
        (e) =>
          ({
            ...e,
            trackId,
          }) as PlayerEventOf<AnyChannelEvent>,
      )
    })
  }
}

export const getLast = <T extends { tick: number }>(
  events: T[],
): T | undefined => maxBy(events, (e) => e.tick)

export const isTickBefore =
  (tick: number) =>
  <T extends { tick: number }>(e: T) =>
    e.tick <= tick

// collect events which will be retained in the synthesizer
const getStatusEvents = (events: TrackEvent[], tick: number) => {
  const controlEvents = events
    .filter(isControllerEvent)
    .filter(isTickBefore(tick))
  // remove duplicated control types
  const recentControlEvents = uniq(controlEvents.map((e) => e.controllerType))
    .map((type) =>
      getLast(controlEvents.filter(isControllerEventWithType(type))),
    )
    .filter(isNotUndefined)

  const setTempo = getLast(
    events.filter(isSetTempoEvent).filter(isTickBefore(tick)),
  )

  const programChange = getLast(
    events.filter(isProgramChangeEvent).filter(isTickBefore(tick)),
  )

  const pitchBend = getLast(
    events.filter(isPitchBendEvent).filter(isTickBefore(tick)),
  )

  return [...recentControlEvents, setTempo, programChange, pitchBend].filter(
    isNotUndefined,
  )
}
