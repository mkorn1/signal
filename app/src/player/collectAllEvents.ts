import { PlayerEvent, PlayerEventOf } from "@signal-app/player"
import { AnyChannelEvent } from "midifile-ts"
import { deassemble as deassembleNote } from "../helpers/noteAssembler"
import Track, { TrackEvent } from "../track"

export const convertTrackEvents = (
  events: TrackEvent[],
  channel: number | undefined,
  trackIndex: number,
) =>
  events
    .filter((e) => !(e.isRecording === true))
    .flatMap((e) => deassembleNote(e))
    .map(
      (e) =>
        ({
          ...e,
          channel: channel,
          trackIndex,
        }) as PlayerEventOf<AnyChannelEvent>,
    )

export const collectAllEvents = (tracks: Track[]): PlayerEvent[] =>
  tracks.flatMap((t, i) => convertTrackEvents(t.events, t.channel, i))
