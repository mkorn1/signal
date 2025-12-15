/**
 * Tool executor for the hybrid agent architecture.
 * Maps backend tool calls to MobX store operations.
 */

import type { NoteEvent, Song, TrackId } from "@signal-app/core"
import {
  controllerMidiEvent,
  emptyTrack,
  isNoteEvent,
  pitchBendMidiEvent,
  timeSignatureMidiEvent,
  toTrackEvents,
} from "@signal-app/core"
import { getControllerNumber } from "../../agent/controllerMapping"
import { getInstrumentProgramNumber } from "../../agent/instrumentMapping"
import {
  generateChordVoicing,
  generateDrumPatternNotes,
  generateBasslineNotes,
  generateMelodyNotes,
  generateArpeggioNotes,
  humanizeNotes,
} from "./musicGeneration"
import {
  generateDrumPatternMagenta,
  generateDrumPatternVAE,
  generateMelodyMagenta,
  generateMelodyVAE,
  generateImprovMagenta,
  isMagentaAvailable,
  type AppNote,
} from "./magentaService"

// ============================================================================
// LOGGING
// ============================================================================

const LOG_PREFIX = "[ToolExecutor]"
const LOG_STYLES = {
  tool: "color: #FF5722; font-weight: bold",
  success: "color: #4CAF50; font-weight: bold",
  fallback: "color: #FF9800; font-weight: bold",
  error: "color: #F44336; font-weight: bold",
}

function logTool(toolName: string, args: Record<string, unknown>) {
  console.log(`%c${LOG_PREFIX} ▶ ${toolName}`, LOG_STYLES.tool, args)
}

function logSuccess(toolName: string, result: Record<string, unknown>) {
  console.log(`%c${LOG_PREFIX} ✓ ${toolName} completed`, LOG_STYLES.success, result)
}

function logFallback(reason: string) {
  console.log(`%c${LOG_PREFIX} ⚠ Falling back to preset: ${reason}`, LOG_STYLES.fallback)
}

function logError(toolName: string, error: unknown) {
  console.error(`%c${LOG_PREFIX} ✗ ${toolName} failed`, LOG_STYLES.error, error)
}

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
async function executeToolCall(song: Song, toolCall: ToolCall): Promise<string> {
  const { name, args } = toolCall
  console.log(
    `%c${LOG_PREFIX} ════════════════════════════════════════`,
    "color: #9C27B0; font-weight: bold"
  )
  console.log(
    `%c${LOG_PREFIX} EXECUTING TOOL: ${name}`,
    "color: #9C27B0; font-weight: bold; font-size: 14px"
  )
  console.log(`%c${LOG_PREFIX} Args:`, "color: #9C27B0", args)

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

      console.log(
        `[HybridAgent] Adding track to song. Current tracks: ${song.tracks.length}`,
      )
      song.addTrack(track)
      const trackId = song.tracks.indexOf(track)
      console.log(
        `[HybridAgent] Track added. New track count: ${song.tracks.length}, trackId: ${trackId}`,
      )

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
      const notes = args.notes as Array<Record<string, unknown>>

      // Validate trackId
      if (typeof trackId !== "number" || isNaN(trackId)) {
        return JSON.stringify({
          error: `Invalid trackId: expected number, got ${typeof trackId}`,
        })
      }

      const track = song.tracks[trackId]
      if (!track) {
        return JSON.stringify({
          error: `Track ${trackId} not found. Available tracks: 0-${song.tracks.length - 1}`,
        })
      }

      // Validate notes array
      if (!Array.isArray(notes)) {
        return JSON.stringify({
          error: `Invalid notes: expected array, got ${typeof notes}`,
        })
      }

      if (notes.length === 0) {
        return JSON.stringify({
          error: `Notes array is empty`,
        })
      }

      // Validate each note and provide helpful error for wrong field names
      const validatedNotes: Array<{
        pitch: number
        start: number
        duration: number
        velocity: number
      }> = []

      for (let i = 0; i < notes.length; i++) {
        const note = notes[i]

        // Check for common field name mistakes
        const pitch = note.pitch ?? note.noteNumber ?? note.note
        const start = note.start ?? note.tick ?? note.position
        const duration = note.duration ?? note.length
        const velocity = note.velocity ?? 100

        if (typeof pitch !== "number" || isNaN(pitch)) {
          return JSON.stringify({
            error: `Note ${i}: invalid pitch. Use "pitch" field (0-127). Got: ${JSON.stringify(note)}`,
          })
        }
        if (typeof start !== "number" || isNaN(start)) {
          return JSON.stringify({
            error: `Note ${i}: invalid start. Use "start" field (ticks). Got: ${JSON.stringify(note)}`,
          })
        }
        if (typeof duration !== "number" || isNaN(duration) || duration <= 0) {
          return JSON.stringify({
            error: `Note ${i}: invalid duration. Use "duration" field (ticks > 0). Got: ${JSON.stringify(note)}`,
          })
        }

        validatedNotes.push({
          pitch: Math.round(pitch),
          start: Math.round(start),
          duration: Math.round(duration),
          velocity: Math.round(Math.max(1, Math.min(127, velocity as number))),
        })
      }

      const noteEvents = validatedNotes.map((note) => ({
        type: "channel" as const,
        subtype: "note" as const,
        noteNumber: note.pitch,
        tick: note.start,
        duration: note.duration,
        velocity: note.velocity,
      }))

      track.addEvents(noteEvents)

      return JSON.stringify({
        trackId,
        noteCount: validatedNotes.length,
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

    // ========================================================================
    // NOTE EDITING TOOLS
    // ========================================================================

    case "deleteNotes": {
      const trackId = args.trackId as number
      const noteIds = args.noteIds as number[]

      const track = song.tracks[trackId]
      if (!track) {
        return JSON.stringify({ error: `Track ${trackId} not found` })
      }

      track.removeEvents(noteIds)

      return JSON.stringify({
        trackId,
        deletedCount: noteIds.length,
      })
    }

    case "updateNotes": {
      const trackId = args.trackId as number
      const updates = args.updates as Array<{
        id: number
        pitch?: number
        tick?: number
        duration?: number
        velocity?: number
      }>

      const track = song.tracks[trackId]
      if (!track) {
        return JSON.stringify({ error: `Track ${trackId} not found` })
      }

      const updateEvents = updates.map((update) => ({
        id: update.id,
        ...(update.pitch !== undefined && { noteNumber: update.pitch }),
        ...(update.tick !== undefined && { tick: update.tick }),
        ...(update.duration !== undefined && { duration: update.duration }),
        ...(update.velocity !== undefined && { velocity: update.velocity }),
      }))

      track.updateEvents(updateEvents)

      return JSON.stringify({
        trackId,
        updatedCount: updates.length,
      })
    }

    case "transposeNotes": {
      const trackId = args.trackId as number
      const noteIds = args.noteIds as number[]
      const semitones = args.semitones as number

      const track = song.tracks[trackId]
      if (!track) {
        return JSON.stringify({ error: `Track ${trackId} not found` })
      }

      // Get notes and transpose them
      const updates = noteIds
        .map((id) => {
          const event = track.getEventById(id)
          if (!event || !isNoteEvent(event)) return null
          const newPitch = Math.max(
            0,
            Math.min(127, event.noteNumber + semitones),
          )
          return { id, noteNumber: newPitch }
        })
        .filter((u): u is { id: number; noteNumber: number } => u !== null)

      track.updateEvents(updates)

      return JSON.stringify({
        trackId,
        transposedCount: updates.length,
        semitones,
      })
    }

    case "duplicateNotes": {
      const trackId = args.trackId as number
      const noteIds = args.noteIds as number[]
      const offsetTicks = (args.offsetTicks as number) ?? 0

      const track = song.tracks[trackId]
      if (!track) {
        return JSON.stringify({ error: `Track ${trackId} not found` })
      }

      // Get the notes to duplicate
      const notesToDuplicate = noteIds
        .map((id) => track.getEventById(id))
        .filter((e): e is NoteEvent => e !== undefined && isNoteEvent(e))

      if (notesToDuplicate.length === 0) {
        return JSON.stringify({
          trackId,
          duplicatedCount: 0,
          newNoteIds: [],
          actualOffset: 0,
        })
      }

      // Calculate offset - if 0, place immediately after the last note
      let actualOffset = offsetTicks
      if (actualOffset === 0) {
        const minTick = Math.min(...notesToDuplicate.map((n) => n.tick))
        const maxEnd = Math.max(
          ...notesToDuplicate.map((n) => n.tick + n.duration),
        )
        actualOffset = maxEnd - minTick
      }

      // Create duplicated notes
      const newNotes = notesToDuplicate.map((note) => ({
        type: "channel" as const,
        subtype: "note" as const,
        noteNumber: note.noteNumber,
        tick: note.tick + actualOffset,
        duration: note.duration,
        velocity: note.velocity,
      }))

      const addedEvents = track.addEvents(newNotes)

      return JSON.stringify({
        trackId,
        duplicatedCount: addedEvents.length,
        newNoteIds: addedEvents.map((e) => e.id),
        actualOffset,
      })
    }

    case "quantizeNotes": {
      const trackId = args.trackId as number
      const noteIds = args.noteIds as number[]
      const gridSize = args.gridSize as number

      const track = song.tracks[trackId]
      if (!track) {
        return JSON.stringify({ error: `Track ${trackId} not found` })
      }

      // Quantize function: round to nearest grid
      const quantize = (tick: number) => Math.round(tick / gridSize) * gridSize

      const updates = noteIds
        .map((id) => {
          const event = track.getEventById(id)
          if (!event || !isNoteEvent(event)) return null
          return { id, tick: quantize(event.tick) }
        })
        .filter((u): u is { id: number; tick: number } => u !== null)

      track.updateEvents(updates)

      return JSON.stringify({
        trackId,
        quantizedCount: updates.length,
      })
    }

    // ========================================================================
    // TRACK OPERATION TOOLS
    // ========================================================================

    case "deleteTrack": {
      const trackId = args.trackId as number

      const track = song.tracks[trackId]
      if (!track) {
        return JSON.stringify({ error: `Track ${trackId} not found` })
      }

      // Don't allow deleting conductor track
      if (track.isConductorTrack) {
        return JSON.stringify({ error: "Cannot delete conductor track" })
      }

      song.removeTrack(track.id as TrackId)

      return JSON.stringify({
        deletedTrackId: trackId,
        success: true,
      })
    }

    case "renameTrack": {
      const trackId = args.trackId as number
      const name = args.name as string

      const track = song.tracks[trackId]
      if (!track) {
        return JSON.stringify({ error: `Track ${trackId} not found` })
      }

      track.setName(name)

      return JSON.stringify({
        trackId,
        newName: name,
      })
    }

    case "setTrackInstrument": {
      const trackId = args.trackId as number
      const instrumentName = args.instrumentName as string

      const track = song.tracks[trackId]
      if (!track) {
        return JSON.stringify({ error: `Track ${trackId} not found` })
      }

      const instrumentInfo = getInstrumentProgramNumber(instrumentName)
      if (!instrumentInfo) {
        return JSON.stringify({
          error: `Unknown instrument: "${instrumentName}"`,
        })
      }

      // Don't change drums to non-drums or vice versa
      if (track.isRhythmTrack && !instrumentInfo.isDrums) {
        return JSON.stringify({
          error: "Cannot change drum track to non-drum instrument",
        })
      }

      if (!track.isRhythmTrack && instrumentInfo.isDrums) {
        return JSON.stringify({
          error:
            "Cannot change non-drum track to drum instrument. Create a new drum track instead.",
        })
      }

      track.setProgramNumber(instrumentInfo.programNumber)
      track.setName(instrumentInfo.instrumentName)

      return JSON.stringify({
        trackId,
        instrumentName: instrumentInfo.instrumentName,
        programNumber: instrumentInfo.programNumber,
      })
    }

    case "setTrackVolume": {
      const trackId = args.trackId as number
      const volume = args.volume as number
      const tick = (args.tick as number) ?? 0

      const track = song.tracks[trackId]
      if (!track) {
        return JSON.stringify({ error: `Track ${trackId} not found` })
      }

      // Clamp volume to valid MIDI range
      const clampedVolume = Math.max(0, Math.min(127, volume))
      track.setVolume(clampedVolume, tick)

      return JSON.stringify({
        trackId,
        volume: clampedVolume,
        tick,
      })
    }

    case "setTrackPan": {
      const trackId = args.trackId as number
      const pan = args.pan as number
      const tick = (args.tick as number) ?? 0

      const track = song.tracks[trackId]
      if (!track) {
        return JSON.stringify({ error: `Track ${trackId} not found` })
      }

      // Clamp pan to valid MIDI range
      const clampedPan = Math.max(0, Math.min(127, pan))
      track.setPan(clampedPan, tick)

      return JSON.stringify({
        trackId,
        pan: clampedPan,
        tick,
      })
    }

    // ========================================================================
    // ADVANCED CONTROLLER TOOLS
    // ========================================================================

    case "setController": {
      const trackId = args.trackId as number
      const controllerType = args.controllerType as string
      const value = args.value as number
      const tick = (args.tick as number) ?? 0

      const track = song.tracks[trackId]
      if (!track) {
        return JSON.stringify({ error: `Track ${trackId} not found` })
      }

      // Resolve controller name/number to CC number
      const controllerInfo = getControllerNumber(controllerType)
      if (!controllerInfo) {
        return JSON.stringify({
          error: `Unknown controller: "${controllerType}". Use names like "sustain", "modulation", "reverb" or CC numbers like "CC64", "1", etc.`,
        })
      }

      // Clamp value to valid MIDI range
      const clampedValue = Math.max(0, Math.min(127, value))

      // Create controller event and add to track
      const [controllerEvent] = toTrackEvents([
        controllerMidiEvent(
          0,
          track.channel ?? 0,
          controllerInfo.controllerNumber,
          clampedValue,
        ),
      ])

      track.addEvent({
        ...controllerEvent,
        tick,
      })

      return JSON.stringify({
        trackId,
        controllerType: controllerInfo.controllerName,
        controllerNumber: controllerInfo.controllerNumber,
        value: clampedValue,
        tick,
      })
    }

    case "setPitchBend": {
      const trackId = args.trackId as number
      const value = args.value as number
      const tick = (args.tick as number) ?? 0

      const track = song.tracks[trackId]
      if (!track) {
        return JSON.stringify({ error: `Track ${trackId} not found` })
      }

      // Clamp value to valid pitch bend range (14-bit: 0-16383)
      const clampedValue = Math.max(0, Math.min(16383, value))

      // Create pitch bend event and add to track
      const [pitchBendEvent] = toTrackEvents([
        pitchBendMidiEvent(0, track.channel ?? 0, clampedValue),
      ])

      track.addEvent({
        ...pitchBendEvent,
        tick,
      })

      return JSON.stringify({
        trackId,
        value: clampedValue,
        tick,
      })
    }

    // ========================================================================
    // HIGH-LEVEL COMPOSITION TOOLS
    // ========================================================================

    case "createChordProgression": {
      const trackId = args.trackId as number
      const chords = args.chords as string[]
      const startTick = (args.startTick as number) ?? 0
      const ticksPerChord = (args.ticksPerChord as number) ?? 1920
      const style = (args.style as string) ?? "block"
      const octave = (args.octave as number) ?? 4
      const velocity = (args.velocity as number) ?? 80

      const track = song.tracks[trackId]
      if (!track) {
        return JSON.stringify({ error: `Track ${trackId} not found` })
      }

      if (!Array.isArray(chords) || chords.length === 0) {
        return JSON.stringify({ error: "chords must be a non-empty array" })
      }

      const allNotes: Array<{
        pitch: number
        start: number
        duration: number
        velocity: number
      }> = []

      chords.forEach((chord, i) => {
        const chordStart = startTick + i * ticksPerChord
        const notes = generateChordVoicing(
          chord,
          octave,
          style,
          chordStart,
          ticksPerChord - 60,
          velocity,
        )
        allNotes.push(...notes)
      })

      const noteEvents = allNotes.map((note) => ({
        type: "channel" as const,
        subtype: "note" as const,
        noteNumber: note.pitch,
        tick: note.start,
        duration: note.duration,
        velocity: note.velocity,
      }))

      const addedEvents = track.addEvents(noteEvents)

      return JSON.stringify({
        trackId,
        noteCount: addedEvents.length,
        chordCount: chords.length,
        style,
      })
    }

    case "generateDrumPattern": {
      const trackId = args.trackId as number
      const style = args.style as string
      const bars = (args.bars as number) ?? 4
      const startTick = (args.startTick as number) ?? 0
      const variation = (args.variation as string) ?? "medium"
      const includeFills = (args.includeFills as boolean) ?? true
      const swing = (args.swing as number) ?? 0
      const temperature = args.temperature as number | undefined
      const useMagenta = (args.useMagenta as boolean) ?? true // Default to Magenta
      const useVAE = (args.useVAE as boolean) ?? false

      logTool("generateDrumPattern", {
        trackId, style, bars, startTick, temperature,
        useMagenta, useVAE, variation, includeFills, swing
      })

      const track = song.tracks[trackId]
      if (!track) {
        logError("generateDrumPattern", `Track ${trackId} not found`)
        return JSON.stringify({ error: `Track ${trackId} not found` })
      }

      if (!track.isRhythmTrack) {
        logError("generateDrumPattern", `Track ${trackId} is not a drum track`)
        return JSON.stringify({
          error: `Track ${trackId} is not a drum track. Use a drum track for generateDrumPattern.`,
        })
      }

      let notes: AppNote[]
      let generationMethod = "preset"

      // Use Magenta if available and requested
      if (useMagenta && isMagentaAvailable()) {
        try {
          console.log(`%c${LOG_PREFIX} Using Magenta AI for drum generation (${useVAE ? 'MusicVAE' : 'DrumsRNN'})`, LOG_STYLES.tool)
          if (useVAE) {
            // MusicVAE - more varied, samples from latent space
            notes = await generateDrumPatternVAE({
              style,
              bars,
              temperature: temperature ?? 1.0,
              startTick,
            })
            generationMethod = "magenta_vae"
          } else {
            // DrumsRNN - continues from seed, style-specific
            notes = await generateDrumPatternMagenta({
              style,
              bars,
              temperature: temperature ?? 1.0,
              startTick,
            })
            generationMethod = "magenta_rnn"
          }
        } catch (err) {
          logFallback(`Magenta error: ${err}`)
          // Fallback to preset patterns
          const hits = generateDrumPatternNotes(
            style,
            bars,
            startTick,
            variation,
            includeFills,
            swing,
          )
          notes = hits.map((h) => ({
            pitch: h.pitch,
            start: h.tick,
            duration: h.duration,
            velocity: h.velocity,
          }))
          generationMethod = "preset_fallback"
        }
      } else {
        logFallback(useMagenta ? "Magenta not available" : "useMagenta=false")
        // Use preset patterns
        const hits = generateDrumPatternNotes(
          style,
          bars,
          startTick,
          variation,
          includeFills,
          swing,
        )
        notes = hits.map((h) => ({
          pitch: h.pitch,
          start: h.tick,
          duration: h.duration,
          velocity: h.velocity,
        }))
      }

      const noteEvents = notes.map((note) => ({
        type: "channel" as const,
        subtype: "note" as const,
        noteNumber: note.pitch,
        tick: note.start,
        duration: note.duration,
        velocity: note.velocity,
      }))

      const addedEvents = track.addEvents(noteEvents)

      const result = {
        trackId,
        noteCount: addedEvents.length,
        bars,
        style,
        generationMethod,
      }
      logSuccess("generateDrumPattern", result)

      return JSON.stringify(result)
    }

    case "generateBassline": {
      const trackId = args.trackId as number
      const chordProgression = args.chordProgression as string[]
      const startTick = (args.startTick as number) ?? 0
      const ticksPerChord = (args.ticksPerChord as number) ?? 1920
      const style = (args.style as string) ?? "root"
      const octave = (args.octave as number) ?? 2
      const velocity = (args.velocity as number) ?? 90

      const track = song.tracks[trackId]
      if (!track) {
        return JSON.stringify({ error: `Track ${trackId} not found` })
      }

      if (!Array.isArray(chordProgression) || chordProgression.length === 0) {
        return JSON.stringify({
          error: "chordProgression must be a non-empty array",
        })
      }

      const bassNotes = generateBasslineNotes(
        chordProgression,
        startTick,
        ticksPerChord,
        style,
        octave,
        velocity,
      )

      const noteEvents = bassNotes.map((note) => ({
        type: "channel" as const,
        subtype: "note" as const,
        noteNumber: note.pitch,
        tick: note.start,
        duration: note.duration,
        velocity: note.velocity,
      }))

      const addedEvents = track.addEvents(noteEvents)

      return JSON.stringify({
        trackId,
        noteCount: addedEvents.length,
        chordCount: chordProgression.length,
        style,
      })
    }

    case "generateMelody": {
      const trackId = args.trackId as number
      const scale = args.scale as string
      const bars = (args.bars as number) ?? 4
      const startTick = (args.startTick as number) ?? 0
      const contour = (args.contour as string) ?? "arch"
      const density = (args.density as string) ?? "medium"
      const rangeLow = (args.range_low as number) ?? 60
      const rangeHigh = (args.range_high as number) ?? 84
      const velocity = (args.velocity as number) ?? 85
      const temperature = args.temperature as number | undefined
      const useMagenta = (args.useMagenta as boolean) ?? true // Default to Magenta
      const useVAE = (args.useVAE as boolean) ?? false
      const chordProgression = args.chordProgression as string[] | undefined

      logTool("generateMelody", {
        trackId, scale, bars, startTick, temperature,
        useMagenta, useVAE, chordProgression,
        rangeLow, rangeHigh, contour, density
      })

      const track = song.tracks[trackId]
      if (!track) {
        logError("generateMelody", `Track ${trackId} not found`)
        return JSON.stringify({ error: `Track ${trackId} not found` })
      }

      let notes: AppNote[]
      let generationMethod = "preset"

      // Use Magenta if available and requested
      if (useMagenta && isMagentaAvailable()) {
        try {
          if (chordProgression && chordProgression.length > 0) {
            // Use ImprovRNN for melody over chords
            console.log(`%c${LOG_PREFIX} Using Magenta ImprovRNN (chord-following melody)`, LOG_STYLES.tool)
            notes = await generateImprovMagenta({
              chordProgression,
              bars,
              temperature: temperature ?? 1.0,
              startTick,
              ticksPerChord: Math.floor((bars * 1920 * 4) / chordProgression.length),
            })
            generationMethod = "magenta_improv"
          } else if (useVAE) {
            // MusicVAE - more varied
            console.log(`%c${LOG_PREFIX} Using Magenta MusicVAE (varied melody)`, LOG_STYLES.tool)
            notes = await generateMelodyVAE({
              scale,
              bars,
              temperature: temperature ?? 1.0,
              startTick,
              rangeLow,
              rangeHigh,
            })
            generationMethod = "magenta_vae"
          } else {
            // MelodyRNN - continues from seed
            console.log(`%c${LOG_PREFIX} Using Magenta MelodyRNN (scale-based melody)`, LOG_STYLES.tool)
            notes = await generateMelodyMagenta({
              scale,
              bars,
              temperature: temperature ?? 1.0,
              startTick,
              rangeLow,
              rangeHigh,
            })
            generationMethod = "magenta_rnn"
          }
        } catch (err) {
          logFallback(`Magenta error: ${err}`)
          // Fallback to preset generation
          notes = generateMelodyNotes(
            scale,
            bars,
            startTick,
            contour,
            density,
            rangeLow,
            rangeHigh,
            velocity,
          )
          generationMethod = "preset_fallback"
        }
      } else {
        logFallback(useMagenta ? "Magenta not available" : "useMagenta=false")
        // Use preset generation
        notes = generateMelodyNotes(
          scale,
          bars,
          startTick,
          contour,
          density,
          rangeLow,
          rangeHigh,
          velocity,
        )
      }

      const noteEvents = notes.map((note) => ({
        type: "channel" as const,
        subtype: "note" as const,
        noteNumber: note.pitch,
        tick: note.start,
        duration: note.duration,
        velocity: note.velocity,
      }))

      const addedEvents = track.addEvents(noteEvents)

      const result = {
        trackId,
        noteCount: addedEvents.length,
        bars,
        scale,
        generationMethod,
      }
      logSuccess("generateMelody", result)

      return JSON.stringify(result)
    }

    case "applyHumanization": {
      const trackId = args.trackId as number
      const noteIds = args.noteIds as number[]
      const velocityVariation = (args.velocityVariation as number) ?? 10
      const timingVariation = (args.timingVariation as number) ?? 10
      const swing = (args.swing as number) ?? 0

      const track = song.tracks[trackId]
      if (!track) {
        return JSON.stringify({ error: `Track ${trackId} not found` })
      }

      // Get notes to humanize
      let notesToProcess: NoteEvent[]
      if (!noteIds || noteIds.length === 0) {
        // Humanize all notes in track
        notesToProcess = track.events.filter(isNoteEvent)
      } else {
        notesToProcess = noteIds
          .map((id) => track.getEventById(id))
          .filter((e): e is NoteEvent => e !== undefined && isNoteEvent(e))
      }

      if (notesToProcess.length === 0) {
        return JSON.stringify({
          trackId,
          humanizedCount: 0,
          message: "No notes to humanize",
        })
      }

      const noteData = notesToProcess.map((n) => ({
        id: n.id,
        tick: n.tick,
        velocity: n.velocity,
        duration: n.duration,
      }))

      const updates = humanizeNotes(noteData, {
        velocityVariation,
        timingVariation,
        swing,
      })

      // Apply updates
      const updateEvents = updates.map((u) => ({
        id: u.id,
        ...(u.tick !== undefined && { tick: u.tick }),
        ...(u.velocity !== undefined && { velocity: u.velocity }),
      }))

      track.updateEvents(updateEvents)

      return JSON.stringify({
        trackId,
        humanizedCount: updates.length,
        velocityVariation,
        timingVariation,
        swing,
      })
    }

    case "createArpeggio": {
      const trackId = args.trackId as number
      const chord = args.chord as string
      const startTick = (args.startTick as number) ?? 0
      const duration = (args.duration as number) ?? 1920
      const pattern = (args.pattern as string) ?? "up"
      const rate = (args.rate as number) ?? 240
      const octaves = (args.octaves as number) ?? 1
      const velocity = (args.velocity as number) ?? 80

      const track = song.tracks[trackId]
      if (!track) {
        return JSON.stringify({ error: `Track ${trackId} not found` })
      }

      const arpeggioNotes = generateArpeggioNotes(
        chord,
        startTick,
        duration,
        pattern,
        rate,
        octaves,
        velocity,
      )

      const noteEvents = arpeggioNotes.map((note) => ({
        type: "channel" as const,
        subtype: "note" as const,
        noteNumber: note.pitch,
        tick: note.start,
        duration: note.duration,
        velocity: note.velocity,
      }))

      const addedEvents = track.addEvents(noteEvents)

      return JSON.stringify({
        trackId,
        noteCount: addedEvents.length,
        chord,
        pattern,
        duration,
      })
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
export async function executeToolCalls(
  song: Song,
  toolCalls: ToolCall[],
): Promise<ToolResult[]> {
  console.log(
    `%c${LOG_PREFIX} ▶▶▶ EXECUTING ${toolCalls.length} TOOL CALLS ▶▶▶`,
    "color: #E91E63; font-weight: bold; font-size: 16px"
  )
  
  const results: ToolResult[] = []
  for (const tc of toolCalls) {
    const result = await executeToolCall(song, tc)
    results.push({ id: tc.id, result })
  }
  
  console.log(
    `%c${LOG_PREFIX} ◀◀◀ ALL ${toolCalls.length} TOOLS COMPLETED ◀◀◀`,
    "color: #E91E63; font-weight: bold; font-size: 16px"
  )
  
  return results
}
