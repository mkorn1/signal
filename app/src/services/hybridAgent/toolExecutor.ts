/**
 * Tool executor for the hybrid agent architecture.
 * Maps backend tool calls to MobX store operations.
 */

import type { Song, TrackEventOf } from "@signal-app/core"
import {
  emptyTrack,
  timeSignatureMidiEvent,
  toTrackEvents,
  isVolumeEvent,
  isPanEvent,
  isControllerEventWithType,
} from "@signal-app/core"
import {
  getInstrumentProgramNumber,
} from "../../agent/instrumentMapping"
import type { ControllerEvent, PitchBendEvent } from "midifile-ts"

export interface ToolCall {
  id: string
  name: string
  args: Record<string, unknown>
}

export interface ToolResult {
  id: string
  result: string // JSON string
}

const DRUM_CHANNEL = 9
const MAX_MIDI_CHANNELS = 16

function getAvailableChannel(song: Song, isDrums: boolean): number {
  if (isDrums) {
    return DRUM_CHANNEL
  }

  const usedChannels = new Set<number>()
  for (const track of song.tracks) {
    if (track.channel !== undefined) {
      usedChannels.add(track.channel)
    }
  }

  for (let ch = 0; ch < MAX_MIDI_CHANNELS; ch++) {
    if (ch === DRUM_CHANNEL) continue
    if (!usedChannels.has(ch)) {
      return ch
    }
  }

  return 0
}

/**
 * Execute a single tool call against the song store.
 */
function executeToolCall(song: Song, toolCall: ToolCall): string {
  const { name, args } = toolCall
  console.log(`[HybridAgent] Executing tool: ${name}`, args)

  switch (name) {
    case "createTrack": {
      const instrumentName = args.instrumentName as string
      const trackName = args.trackName as string | undefined

      const instrumentInfo = getInstrumentProgramNumber(instrumentName)
      if (!instrumentInfo) {
        console.error(`[HybridAgent] Unknown instrument: ${instrumentName}`)
        return JSON.stringify({
          error: `Unknown instrument: "${instrumentName}"`,
        })
      }

      const channel = getAvailableChannel(song, instrumentInfo.isDrums)
      const track = emptyTrack(channel)
      track.setName(trackName ?? instrumentInfo.instrumentName)

      if (!instrumentInfo.isDrums) {
        track.setProgramNumber(instrumentInfo.programNumber)
      }

      console.log(`[HybridAgent] Adding track to song. Current tracks: ${song.tracks.length}`)
      song.addTrack(track)
      const trackId = song.tracks.indexOf(track)
      console.log(`[HybridAgent] Track added. New track count: ${song.tracks.length}, trackId: ${trackId}`)

      return JSON.stringify({
        trackId,
        instrumentName: instrumentInfo.instrumentName,
        programNumber: instrumentInfo.programNumber,
        channel,
        isDrums: instrumentInfo.isDrums,
      })
    }

    case "addNotes": {
      const trackId = args.trackId as number
      const notes = args.notes as Array<{
        pitch: number
        start: number
        duration: number
        velocity?: number
      }>

      const track = song.tracks[trackId]
      if (!track) {
        return JSON.stringify({
          error: `Track ${trackId} not found`,
        })
      }

      const noteEvents = notes.map((note) => ({
        type: "channel" as const,
        subtype: "note" as const,
        noteNumber: note.pitch,
        tick: note.start,
        duration: note.duration,
        velocity: note.velocity ?? 100,
      }))

      track.addEvents(noteEvents)

      return JSON.stringify({
        trackId,
        noteCount: notes.length,
      })
    }

    case "setTempo": {
      const bpm = args.bpm as number
      const tick = (args.tick as number) ?? 0

      const conductor = song.conductorTrack
      if (!conductor) {
        return JSON.stringify({
          error: "No conductor track found",
        })
      }

      conductor.setTempo(bpm, tick)

      return JSON.stringify({ bpm, tick })
    }

    case "setTimeSignature": {
      const numerator = args.numerator as number
      const denominator = args.denominator as number
      const tick = (args.tick as number) ?? 0

      const conductor = song.conductorTrack
      if (!conductor) {
        return JSON.stringify({
          error: "No conductor track found",
        })
      }

      const [tsEvent] = toTrackEvents([
        timeSignatureMidiEvent(0, numerator, denominator),
      ])

      conductor.addEvent({
        ...tsEvent,
        tick,
      })

      return JSON.stringify({ numerator, denominator, tick })
    }

    case "addEffects": {
      const trackId = args.trackId as number
      const effects = args.effects as Array<{
        effect_type: "volume" | "pan" | "program_change" | "expression" | "pitch_bend" | "sustain"
        value: number
        tick?: number
      }>

      const track = song.tracks[trackId]
      if (!track) {
        return JSON.stringify({
          error: `Track ${trackId} not found`,
        })
      }

      // Validate effects
      const errors: string[] = []
      for (const effect of effects) {
        if (effect.tick !== undefined && effect.tick < 0) {
          errors.push(`Invalid tick ${effect.tick}. Must be >= 0.`)
        }
        if (effect.effect_type === "pitch_bend") {
          if (effect.value < 0 || effect.value > 16384) {
            errors.push(
              `Invalid pitch_bend value ${effect.value}. Must be 0-16384 (8192 = center).`,
            )
          }
        } else {
          if (effect.value < 0 || effect.value > 127) {
            errors.push(
              `Invalid ${effect.effect_type} value ${effect.value}. Must be 0-127.`,
            )
          }
        }
      }

      if (errors.length > 0) {
        return JSON.stringify({
          error: errors.join("; "),
        })
      }

      // Apply effects
      let effectsAdded = 0
      for (const effect of effects) {
        try {
          const tick = effect.tick ?? 0
          if (effect.effect_type === "volume") {
            // If tick is 0 (default), remove all existing volume automation first
            // This ensures the new volume applies to the entire track
            if (tick === 0) {
              const volumeEvents = track.events.filter(isVolumeEvent)
              if (volumeEvents.length > 0) {
                track.removeEvents(volumeEvents.map((e) => e.id))
              }
            }
            track.setVolume(Math.round(effect.value), tick)
            effectsAdded++
          } else if (effect.effect_type === "pan") {
            // If tick is 0 (default), remove all existing pan automation first
            if (tick === 0) {
              const panEvents = track.events.filter(isPanEvent)
              if (panEvents.length > 0) {
                track.removeEvents(panEvents.map((e) => e.id))
              }
            }
            track.setPan(Math.round(effect.value), tick)
            effectsAdded++
          } else if (effect.effect_type === "program_change") {
            track.setProgramNumber(Math.round(effect.value))
            effectsAdded++
          } else if (effect.effect_type === "expression") {
            // Expression is CC11
            track.createOrUpdate<TrackEventOf<ControllerEvent>>({
              type: "channel",
              subtype: "controller",
              controllerType: 11,
              tick,
              value: Math.round(effect.value),
            })
            effectsAdded++
          } else if (effect.effect_type === "sustain") {
            // Sustain/Hold Pedal is CC64
            track.createOrUpdate<TrackEventOf<ControllerEvent>>({
              type: "channel",
              subtype: "controller",
              controllerType: 64,
              tick,
              value: Math.round(effect.value),
            })
            effectsAdded++
          } else if (effect.effect_type === "pitch_bend") {
            // Pitch bend is a special event type, not a controller
            track.createOrUpdate<TrackEventOf<PitchBendEvent>>({
              type: "channel",
              subtype: "pitchBend",
              tick,
              value: Math.round(effect.value),
            })
            effectsAdded++
          }
        } catch (error) {
          errors.push(
            `Failed to add ${effect.effect_type}: ${error instanceof Error ? error.message : "Unknown error"}`,
          )
        }
      }

      if (errors.length > 0 && effectsAdded === 0) {
        return JSON.stringify({
          error: errors.join("; "),
        })
      }

      // Return success - include errors only if some effects failed (partial success)
      const result: { trackId: number; effectsAdded: number; errors?: string[] } = {
        trackId,
        effectsAdded,
      }
      if (errors.length > 0) {
        result.errors = errors
      }
      return JSON.stringify(result)
    }

    default:
      return JSON.stringify({
        error: `Unknown tool: ${name}`,
      })
  }
}

/**
 * Execute multiple tool calls and return results.
 */
export function executeToolCalls(
  song: Song,
  toolCalls: ToolCall[]
): ToolResult[] {
  return toolCalls.map((tc) => ({
    id: tc.id,
    result: executeToolCall(song, tc),
  }))
}
