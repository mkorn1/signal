/**
 * Music generation utilities for high-level composition tools.
 * Contains chord parsing, pattern generation, and humanization logic.
 */

// ============================================================================
// CHORD PARSING AND VOICING
// ============================================================================

const NOTE_MAP: Record<string, number> = {
  C: 0,
  "C#": 1,
  Db: 1,
  D: 2,
  "D#": 3,
  Eb: 3,
  E: 4,
  F: 5,
  "F#": 6,
  Gb: 6,
  G: 7,
  "G#": 8,
  Ab: 8,
  A: 9,
  "A#": 10,
  Bb: 10,
  B: 11,
}

interface ChordTones {
  root: number
  intervals: number[]
  bass?: number // For slash chords
}

/**
 * Parse a chord symbol into its component intervals.
 */
export function parseChord(chordSymbol: string): ChordTones | null {
  // Handle slash chords first (e.g., "C/E")
  let bass: number | undefined
  let mainChord = chordSymbol
  if (chordSymbol.includes("/")) {
    const [chord, bassNote] = chordSymbol.split("/")
    mainChord = chord
    bass = NOTE_MAP[bassNote]
    if (bass === undefined) return null
  }

  // Extract root note
  let rootMatch = mainChord.match(/^([A-G][#b]?)/)
  if (!rootMatch) return null

  const rootName = rootMatch[1]
  const root = NOTE_MAP[rootName]
  if (root === undefined) return null

  // Get quality and extensions
  const quality = mainChord.slice(rootName.length)

  // Define interval patterns for different chord types
  let intervals: number[]

  // Major variations
  if (
    quality === "" ||
    quality === "maj" ||
    quality === "M" ||
    quality === "major"
  ) {
    intervals = [0, 4, 7] // Major triad
  } else if (quality === "maj7" || quality === "M7" || quality === "Δ7") {
    intervals = [0, 4, 7, 11] // Major 7th
  } else if (quality === "maj9" || quality === "M9" || quality === "Δ9") {
    intervals = [0, 4, 7, 11, 14] // Major 9th
  } else if (quality === "6" || quality === "maj6") {
    intervals = [0, 4, 7, 9] // Major 6th
  } else if (quality === "add9") {
    intervals = [0, 4, 7, 14] // Add9
  }
  // Minor variations
  else if (
    quality === "m" ||
    quality === "min" ||
    quality === "-" ||
    quality === "minor"
  ) {
    intervals = [0, 3, 7] // Minor triad
  } else if (
    quality === "m7" ||
    quality === "min7" ||
    quality === "-7" ||
    quality === "mi7"
  ) {
    intervals = [0, 3, 7, 10] // Minor 7th
  } else if (quality === "m9" || quality === "min9" || quality === "-9") {
    intervals = [0, 3, 7, 10, 14] // Minor 9th
  } else if (quality === "m6" || quality === "min6") {
    intervals = [0, 3, 7, 9] // Minor 6th
  } else if (quality === "mMaj7" || quality === "m(maj7)" || quality === "-Δ7") {
    intervals = [0, 3, 7, 11] // Minor major 7th
  }
  // Dominant variations
  else if (quality === "7" || quality === "dom7") {
    intervals = [0, 4, 7, 10] // Dominant 7th
  } else if (quality === "9") {
    intervals = [0, 4, 7, 10, 14] // Dominant 9th
  } else if (quality === "11") {
    intervals = [0, 4, 7, 10, 14, 17] // Dominant 11th
  } else if (quality === "13") {
    intervals = [0, 4, 7, 10, 14, 21] // Dominant 13th
  } else if (quality === "7#9" || quality === "7(#9)") {
    intervals = [0, 4, 7, 10, 15] // 7#9 (Hendrix chord)
  } else if (quality === "7b9" || quality === "7(b9)") {
    intervals = [0, 4, 7, 10, 13] // 7b9
  } else if (quality === "7#11" || quality === "7(#11)") {
    intervals = [0, 4, 7, 10, 18] // 7#11
  } else if (quality === "7alt") {
    intervals = [0, 4, 8, 10, 13] // Altered dominant
  }
  // Diminished and augmented
  else if (quality === "dim" || quality === "°") {
    intervals = [0, 3, 6] // Diminished triad
  } else if (quality === "dim7" || quality === "°7") {
    intervals = [0, 3, 6, 9] // Diminished 7th
  } else if (
    quality === "m7b5" ||
    quality === "ø" ||
    quality === "ø7" ||
    quality === "half-dim"
  ) {
    intervals = [0, 3, 6, 10] // Half diminished
  } else if (quality === "aug" || quality === "+" || quality === "#5") {
    intervals = [0, 4, 8] // Augmented triad
  } else if (quality === "aug7" || quality === "+7" || quality === "7#5") {
    intervals = [0, 4, 8, 10] // Augmented 7th
  }
  // Sus chords
  else if (quality === "sus4" || quality === "sus") {
    intervals = [0, 5, 7] // Sus4
  } else if (quality === "sus2") {
    intervals = [0, 2, 7] // Sus2
  } else if (quality === "7sus4" || quality === "7sus") {
    intervals = [0, 5, 7, 10] // 7sus4
  }
  // Power chord
  else if (quality === "5" || quality === "power") {
    intervals = [0, 7] // Power chord
  }
  // Default to major if unknown
  else {
    intervals = [0, 4, 7]
  }

  return { root, intervals, bass }
}

/**
 * Generate chord voicing notes.
 */
export function generateChordVoicing(
  chordSymbol: string,
  octave: number,
  style: string,
  startTick: number,
  duration: number,
  velocity: number,
): Array<{ pitch: number; start: number; duration: number; velocity: number }> {
  const chord = parseChord(chordSymbol)
  if (!chord) return []

  const baseNote = chord.root + (octave + 1) * 12
  const notes: Array<{
    pitch: number
    start: number
    duration: number
    velocity: number
  }> = []

  switch (style) {
    case "block": {
      // All notes at once
      for (const interval of chord.intervals) {
        notes.push({
          pitch: baseNote + interval,
          start: startTick,
          duration,
          velocity,
        })
      }
      // Add bass note if slash chord
      if (chord.bass !== undefined) {
        notes.push({
          pitch: chord.bass + octave * 12,
          start: startTick,
          duration,
          velocity,
        })
      }
      break
    }

    case "arpeggiated": {
      // Notes played sequentially
      const noteSpacing = Math.floor(duration / (chord.intervals.length + 1))
      chord.intervals.forEach((interval, i) => {
        notes.push({
          pitch: baseNote + interval,
          start: startTick + i * noteSpacing,
          duration: duration - i * noteSpacing,
          velocity: velocity - i * 5, // Slight decay
        })
      })
      break
    }

    case "broken": {
      // Alternating bass and upper
      const halfDuration = Math.floor(duration / 2)
      // Bass note
      notes.push({
        pitch: baseNote,
        start: startTick,
        duration: halfDuration,
        velocity,
      })
      // Upper notes
      for (let i = 1; i < chord.intervals.length; i++) {
        notes.push({
          pitch: baseNote + chord.intervals[i],
          start: startTick + halfDuration,
          duration: halfDuration,
          velocity: velocity - 10,
        })
      }
      break
    }

    case "spread": {
      // Wide voicing across octaves
      chord.intervals.forEach((interval, i) => {
        const octaveOffset = i % 2 === 0 ? 0 : 12
        notes.push({
          pitch: baseNote + interval + octaveOffset,
          start: startTick,
          duration,
          velocity,
        })
      })
      break
    }

    default:
      // Default to block
      for (const interval of chord.intervals) {
        notes.push({
          pitch: baseNote + interval,
          start: startTick,
          duration,
          velocity,
        })
      }
  }

  return notes
}

// ============================================================================
// DRUM PATTERN GENERATION
// ============================================================================

// General MIDI drum map
const DRUMS = {
  kick: 36,
  snare: 38,
  sideStick: 37,
  clap: 39,
  hihatClosed: 42,
  hihatOpen: 46,
  hihatPedal: 44,
  crash: 49,
  ride: 51,
  rideBell: 53,
  tomLow: 45,
  tomMid: 47,
  tomHigh: 50,
  cowbell: 56,
  tambourine: 54,
  conga: 63,
  bongo: 61,
}

interface DrumHit {
  pitch: number
  tick: number
  velocity: number
  duration: number
}

/**
 * Generate a drum pattern based on style.
 */
export function generateDrumPatternNotes(
  style: string,
  bars: number,
  startTick: number,
  variation: string,
  includeFills: boolean,
  swing: number,
): DrumHit[] {
  const hits: DrumHit[] = []
  const ticksPerBar = 1920
  const ticksPerBeat = 480

  // Variation affects velocity range and ghost notes
  const variationAmount =
    variation === "minimal"
      ? 0
      : variation === "low"
        ? 5
        : variation === "medium"
          ? 10
          : 20

  const swingOffset = Math.floor((swing / 100) * 60) // Max 60 ticks of swing

  for (let bar = 0; bar < bars; bar++) {
    const barStart = startTick + bar * ticksPerBar
    const isFillBar = includeFills && (bar + 1) % 4 === 0 // Fill every 4 bars

    switch (style) {
      case "rock":
        generateRockPattern(
          hits,
          barStart,
          variationAmount,
          swingOffset,
          isFillBar,
        )
        break
      case "pop":
        generatePopPattern(
          hits,
          barStart,
          variationAmount,
          swingOffset,
          isFillBar,
        )
        break
      case "jazz":
        generateJazzPattern(
          hits,
          barStart,
          variationAmount,
          swingOffset,
          isFillBar,
        )
        break
      case "funk":
        generateFunkPattern(
          hits,
          barStart,
          variationAmount,
          swingOffset,
          isFillBar,
        )
        break
      case "hiphop":
        generateHipHopPattern(
          hits,
          barStart,
          variationAmount,
          swingOffset,
          isFillBar,
        )
        break
      case "latin":
        generateLatinPattern(
          hits,
          barStart,
          variationAmount,
          swingOffset,
          isFillBar,
        )
        break
      case "ballad":
        generateBalladPattern(
          hits,
          barStart,
          variationAmount,
          swingOffset,
          isFillBar,
        )
        break
      case "metal":
        generateMetalPattern(
          hits,
          barStart,
          variationAmount,
          swingOffset,
          isFillBar,
        )
        break
      case "electronic":
        generateElectronicPattern(
          hits,
          barStart,
          variationAmount,
          swingOffset,
          isFillBar,
        )
        break
      default:
        generateRockPattern(
          hits,
          barStart,
          variationAmount,
          swingOffset,
          isFillBar,
        )
    }
  }

  return hits
}

function addHit(
  hits: DrumHit[],
  pitch: number,
  tick: number,
  baseVelocity: number,
  variation: number,
): void {
  const velocityVar = Math.floor(Math.random() * variation * 2) - variation
  hits.push({
    pitch,
    tick,
    velocity: Math.max(30, Math.min(127, baseVelocity + velocityVar)),
    duration: 120,
  })
}

function generateRockPattern(
  hits: DrumHit[],
  barStart: number,
  variation: number,
  swing: number,
  isFill: boolean,
): void {
  const beat = 480

  if (isFill) {
    // Tom fill
    addHit(hits, DRUMS.tomHigh, barStart + beat * 2, 100, variation)
    addHit(hits, DRUMS.tomHigh, barStart + beat * 2 + 240, 95, variation)
    addHit(hits, DRUMS.tomMid, barStart + beat * 3, 100, variation)
    addHit(hits, DRUMS.tomMid, barStart + beat * 3 + 240, 95, variation)
    addHit(hits, DRUMS.tomLow, barStart + beat * 3 + 360, 105, variation)
    addHit(hits, DRUMS.crash, barStart + beat * 4, 110, variation)
  } else {
    // Kick on 1 and 3
    addHit(hits, DRUMS.kick, barStart, 100, variation)
    addHit(hits, DRUMS.kick, barStart + beat * 2, 95, variation)

    // Snare on 2 and 4
    addHit(hits, DRUMS.snare, barStart + beat, 100, variation)
    addHit(hits, DRUMS.snare, barStart + beat * 3, 100, variation)

    // Hi-hats on eighth notes
    for (let i = 0; i < 8; i++) {
      const isOffbeat = i % 2 === 1
      const tick = barStart + i * 240 + (isOffbeat ? swing : 0)
      const vel = isOffbeat ? 70 : 85
      addHit(hits, DRUMS.hihatClosed, tick, vel, variation)
    }
  }
}

function generatePopPattern(
  hits: DrumHit[],
  barStart: number,
  variation: number,
  swing: number,
  isFill: boolean,
): void {
  const beat = 480

  // Steady kick pattern
  addHit(hits, DRUMS.kick, barStart, 100, variation)
  addHit(hits, DRUMS.kick, barStart + beat, 90, variation)
  addHit(hits, DRUMS.kick, barStart + beat * 2, 95, variation)
  addHit(hits, DRUMS.kick, barStart + beat * 3, 90, variation)

  // Snare/clap on 2 and 4
  addHit(hits, DRUMS.snare, barStart + beat, 100, variation)
  addHit(hits, DRUMS.clap, barStart + beat, 80, variation)
  addHit(hits, DRUMS.snare, barStart + beat * 3, 100, variation)
  addHit(hits, DRUMS.clap, barStart + beat * 3, 80, variation)

  // Hi-hats
  for (let i = 0; i < 8; i++) {
    const isOffbeat = i % 2 === 1
    const tick = barStart + i * 240 + (isOffbeat ? swing : 0)
    addHit(hits, DRUMS.hihatClosed, tick, isOffbeat ? 65 : 80, variation)
  }

  if (isFill) {
    addHit(hits, DRUMS.crash, barStart + beat * 4, 110, variation)
  }
}

function generateJazzPattern(
  hits: DrumHit[],
  barStart: number,
  variation: number,
  swing: number,
  isFill: boolean,
): void {
  const beat = 480

  // Ride cymbal pattern (swing feel)
  for (let i = 0; i < 4; i++) {
    const beatStart = barStart + i * beat
    addHit(hits, DRUMS.ride, beatStart, 80, variation)
    addHit(hits, DRUMS.ride, beatStart + 320 + swing, 60, variation) // Swung eighth
  }

  // Hi-hat on 2 and 4 (foot)
  addHit(hits, DRUMS.hihatPedal, barStart + beat, 70, variation)
  addHit(hits, DRUMS.hihatPedal, barStart + beat * 3, 70, variation)

  // Light kick comping
  if (Math.random() > 0.5) {
    addHit(hits, DRUMS.kick, barStart, 70, variation)
  }
  if (Math.random() > 0.6) {
    addHit(hits, DRUMS.kick, barStart + beat * 2 + 240, 65, variation)
  }

  // Ghost snares
  if (Math.random() > 0.7) {
    addHit(hits, DRUMS.snare, barStart + beat + 240, 40, variation)
  }
}

function generateFunkPattern(
  hits: DrumHit[],
  barStart: number,
  variation: number,
  swing: number,
  isFill: boolean,
): void {
  const sixteenth = 120

  // Syncopated kick pattern
  addHit(hits, DRUMS.kick, barStart, 100, variation)
  addHit(hits, DRUMS.kick, barStart + sixteenth * 6, 95, variation)
  addHit(hits, DRUMS.kick, barStart + sixteenth * 10, 90, variation)

  // Snare on 2 and 4 with ghost notes
  addHit(hits, DRUMS.snare, barStart + sixteenth * 4, 100, variation)
  addHit(hits, DRUMS.snare, barStart + sixteenth * 12, 100, variation)

  // Ghost notes
  addHit(hits, DRUMS.snare, barStart + sixteenth * 3, 40, variation)
  addHit(hits, DRUMS.snare, barStart + sixteenth * 7, 35, variation)
  addHit(hits, DRUMS.snare, barStart + sixteenth * 11, 40, variation)
  addHit(hits, DRUMS.snare, barStart + sixteenth * 15, 35, variation)

  // Hi-hats on sixteenths
  for (let i = 0; i < 16; i++) {
    const isAccent = i % 4 === 0
    const vel = isAccent ? 85 : 55
    addHit(hits, DRUMS.hihatClosed, barStart + i * sixteenth, vel, variation)
  }
}

function generateHipHopPattern(
  hits: DrumHit[],
  barStart: number,
  variation: number,
  swing: number,
  isFill: boolean,
): void {
  const sixteenth = 120

  // Boom bap kick pattern
  addHit(hits, DRUMS.kick, barStart, 110, variation)
  addHit(hits, DRUMS.kick, barStart + sixteenth * 5, 100, variation)
  addHit(hits, DRUMS.kick, barStart + sixteenth * 9, 105, variation)

  // Hard snare on 2 and 4
  addHit(hits, DRUMS.snare, barStart + sixteenth * 4, 110, variation)
  addHit(hits, DRUMS.snare, barStart + sixteenth * 12, 110, variation)

  // Hi-hats with swing
  for (let i = 0; i < 8; i++) {
    const isOffbeat = i % 2 === 1
    const tick = barStart + i * 240 + (isOffbeat ? swing : 0)
    const isOpen = i === 3 || i === 7
    addHit(
      hits,
      isOpen ? DRUMS.hihatOpen : DRUMS.hihatClosed,
      tick,
      isOffbeat ? 60 : 80,
      variation,
    )
  }
}

function generateLatinPattern(
  hits: DrumHit[],
  barStart: number,
  variation: number,
  swing: number,
  isFill: boolean,
): void {
  const sixteenth = 120

  // Clave-influenced kick
  addHit(hits, DRUMS.kick, barStart, 90, variation)
  addHit(hits, DRUMS.kick, barStart + sixteenth * 6, 85, variation)
  addHit(hits, DRUMS.kick, barStart + sixteenth * 10, 90, variation)

  // Side stick pattern
  addHit(hits, DRUMS.sideStick, barStart + sixteenth * 4, 80, variation)
  addHit(hits, DRUMS.sideStick, barStart + sixteenth * 12, 80, variation)

  // Conga pattern
  addHit(hits, DRUMS.conga, barStart + sixteenth * 2, 70, variation)
  addHit(hits, DRUMS.conga, barStart + sixteenth * 5, 75, variation)
  addHit(hits, DRUMS.conga, barStart + sixteenth * 8, 70, variation)
  addHit(hits, DRUMS.conga, barStart + sixteenth * 11, 75, variation)
  addHit(hits, DRUMS.conga, barStart + sixteenth * 14, 70, variation)

  // Hi-hat on quarter notes
  for (let i = 0; i < 4; i++) {
    addHit(hits, DRUMS.hihatClosed, barStart + i * 480, 75, variation)
  }
}

function generateBalladPattern(
  hits: DrumHit[],
  barStart: number,
  variation: number,
  swing: number,
  isFill: boolean,
): void {
  const beat = 480

  // Sparse kick
  addHit(hits, DRUMS.kick, barStart, 85, variation)
  addHit(hits, DRUMS.kick, barStart + beat * 2 + 240, 75, variation)

  // Soft snare
  addHit(hits, DRUMS.snare, barStart + beat, 80, variation)
  addHit(hits, DRUMS.snare, barStart + beat * 3, 80, variation)

  // Ride or hi-hat quarters
  for (let i = 0; i < 4; i++) {
    addHit(hits, DRUMS.ride, barStart + i * beat, 65, variation)
  }
}

function generateMetalPattern(
  hits: DrumHit[],
  barStart: number,
  variation: number,
  swing: number,
  isFill: boolean,
): void {
  const sixteenth = 120

  // Double kick pattern
  for (let i = 0; i < 16; i++) {
    if (i % 2 === 0 || Math.random() > 0.3) {
      addHit(hits, DRUMS.kick, barStart + i * sixteenth, 110, variation)
    }
  }

  // Hard snare
  addHit(hits, DRUMS.snare, barStart + sixteenth * 4, 120, variation)
  addHit(hits, DRUMS.snare, barStart + sixteenth * 12, 120, variation)

  // China/crash accents
  addHit(hits, DRUMS.crash, barStart, 100, variation)

  // Hi-hats
  for (let i = 0; i < 8; i++) {
    addHit(hits, DRUMS.hihatClosed, barStart + i * 240, 90, variation)
  }
}

function generateElectronicPattern(
  hits: DrumHit[],
  barStart: number,
  variation: number,
  swing: number,
  isFill: boolean,
): void {
  const beat = 480

  // Four on the floor kick
  for (let i = 0; i < 4; i++) {
    addHit(hits, DRUMS.kick, barStart + i * beat, 110, variation)
  }

  // Clap on 2 and 4
  addHit(hits, DRUMS.clap, barStart + beat, 100, variation)
  addHit(hits, DRUMS.clap, barStart + beat * 3, 100, variation)

  // Offbeat hi-hats
  for (let i = 0; i < 4; i++) {
    addHit(hits, DRUMS.hihatOpen, barStart + i * beat + 240, 80, variation)
  }
}

// ============================================================================
// BASSLINE GENERATION
// ============================================================================

/**
 * Generate bassline notes following a chord progression.
 */
export function generateBasslineNotes(
  chords: string[],
  startTick: number,
  ticksPerChord: number,
  style: string,
  octave: number,
  velocity: number,
): Array<{ pitch: number; start: number; duration: number; velocity: number }> {
  const notes: Array<{
    pitch: number
    start: number
    duration: number
    velocity: number
  }> = []

  chords.forEach((chordSymbol, i) => {
    const chord = parseChord(chordSymbol)
    if (!chord) return

    const chordStart = startTick + i * ticksPerChord
    const root = chord.bass ?? chord.root
    const basePitch = root + (octave + 1) * 12

    switch (style) {
      case "root":
        // Simple root notes
        notes.push({
          pitch: basePitch,
          start: chordStart,
          duration: ticksPerChord - 60,
          velocity,
        })
        break

      case "fifth":
        // Root and fifth
        notes.push({
          pitch: basePitch,
          start: chordStart,
          duration: ticksPerChord / 2 - 30,
          velocity,
        })
        notes.push({
          pitch: basePitch + 7,
          start: chordStart + ticksPerChord / 2,
          duration: ticksPerChord / 2 - 30,
          velocity: velocity - 5,
        })
        break

      case "walking": {
        // Walking bass with chromatic approaches
        const beatDuration = 480
        const numBeats = Math.floor(ticksPerChord / beatDuration)

        for (let beat = 0; beat < numBeats; beat++) {
          let pitch = basePitch
          if (beat === 0) {
            pitch = basePitch // Root
          } else if (beat === 1) {
            pitch = basePitch + (chord.intervals[1] ?? 4) // Third
          } else if (beat === 2) {
            pitch = basePitch + 7 // Fifth
          } else {
            // Approach note to next chord
            const nextChord = chords[(i + 1) % chords.length]
            const nextParsed = parseChord(nextChord)
            if (nextParsed) {
              const nextRoot = (nextParsed.bass ?? nextParsed.root) + 60
              pitch = nextRoot > basePitch ? basePitch + 11 : basePitch - 1
            } else {
              pitch = basePitch + 5
            }
          }
          notes.push({
            pitch,
            start: chordStart + beat * beatDuration,
            duration: beatDuration - 60,
            velocity: velocity - (beat % 2) * 10,
          })
        }
        break
      }

      case "arpeggiated": {
        // Arpeggiate chord tones
        const noteCount = Math.min(chord.intervals.length, 4)
        const noteDuration = Math.floor(ticksPerChord / noteCount)
        for (let n = 0; n < noteCount; n++) {
          notes.push({
            pitch: basePitch + chord.intervals[n % chord.intervals.length],
            start: chordStart + n * noteDuration,
            duration: noteDuration - 30,
            velocity: velocity - n * 5,
          })
        }
        break
      }

      case "syncopated": {
        // Syncopated funk-style
        const sixteenth = 120
        notes.push({
          pitch: basePitch,
          start: chordStart,
          duration: sixteenth * 2,
          velocity,
        })
        notes.push({
          pitch: basePitch,
          start: chordStart + sixteenth * 3,
          duration: sixteenth,
          velocity: velocity - 10,
        })
        notes.push({
          pitch: basePitch + 7,
          start: chordStart + sixteenth * 6,
          duration: sixteenth * 2,
          velocity: velocity - 5,
        })
        notes.push({
          pitch: basePitch,
          start: chordStart + sixteenth * 10,
          duration: sixteenth * 2,
          velocity: velocity - 10,
        })
        break
      }

      case "octave": {
        // Root with octave
        const half = ticksPerChord / 2
        notes.push({
          pitch: basePitch,
          start: chordStart,
          duration: half - 30,
          velocity,
        })
        notes.push({
          pitch: basePitch + 12,
          start: chordStart + half,
          duration: half - 30,
          velocity: velocity - 5,
        })
        break
      }

      case "pedal":
        // Sustained pedal tone
        notes.push({
          pitch: basePitch,
          start: chordStart,
          duration: ticksPerChord,
          velocity,
        })
        break

      default:
        notes.push({
          pitch: basePitch,
          start: chordStart,
          duration: ticksPerChord - 60,
          velocity,
        })
    }
  })

  return notes
}

// ============================================================================
// MELODY GENERATION
// ============================================================================

interface ScaleInfo {
  intervals: number[]
}

const SCALES: Record<string, number[]> = {
  major: [0, 2, 4, 5, 7, 9, 11],
  minor: [0, 2, 3, 5, 7, 8, 10],
  dorian: [0, 2, 3, 5, 7, 9, 10],
  phrygian: [0, 1, 3, 5, 7, 8, 10],
  lydian: [0, 2, 4, 6, 7, 9, 11],
  mixolydian: [0, 2, 4, 5, 7, 9, 10],
  pentatonic: [0, 2, 4, 7, 9],
  blues: [0, 3, 5, 6, 7, 10],
  chromatic: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
}

function parseScale(scaleStr: string): { root: number; intervals: number[] } {
  // Parse scale like "A minor", "C major", "D dorian"
  const parts = scaleStr.trim().split(/\s+/)
  let rootStr = parts[0]
  let scaleType = parts.slice(1).join(" ").toLowerCase() || "major"

  // Handle shorthand like "Am", "Cm"
  if (rootStr.endsWith("m") && rootStr.length > 1) {
    rootStr = rootStr.slice(0, -1)
    scaleType = "minor"
  }

  const root = NOTE_MAP[rootStr] ?? 0

  let intervals = SCALES[scaleType]
  if (!intervals) {
    // Try to find partial match
    for (const [key, val] of Object.entries(SCALES)) {
      if (scaleType.includes(key)) {
        intervals = val
        break
      }
    }
  }
  intervals = intervals ?? SCALES.major

  return { root, intervals }
}

/**
 * Generate melody notes.
 */
export function generateMelodyNotes(
  scale: string,
  bars: number,
  startTick: number,
  contour: string,
  density: string,
  rangeLow: number,
  rangeHigh: number,
  velocity: number,
): Array<{ pitch: number; start: number; duration: number; velocity: number }> {
  const notes: Array<{
    pitch: number
    start: number
    duration: number
    velocity: number
  }> = []
  const ticksPerBar = 1920
  const totalTicks = bars * ticksPerBar

  const { root, intervals } = parseScale(scale)

  // Get all scale notes in range
  const scaleNotes: number[] = []
  for (let oct = 0; oct < 10; oct++) {
    for (const interval of intervals) {
      const pitch = root + oct * 12 + interval
      if (pitch >= rangeLow && pitch <= rangeHigh) {
        scaleNotes.push(pitch)
      }
    }
  }

  if (scaleNotes.length === 0) return notes

  // Determine note count based on density
  const densityMultiplier =
    density === "sparse" ? 0.5 : density === "dense" ? 2 : 1
  const notesPerBar = Math.round(4 * densityMultiplier)
  const totalNotes = notesPerBar * bars

  // Durations based on density
  const durations =
    density === "sparse"
      ? [960, 720, 480]
      : density === "dense"
        ? [240, 120, 360]
        : [480, 240, 720, 360]

  // Generate contour curve
  const contourValues: number[] = []
  for (let i = 0; i < totalNotes; i++) {
    const t = i / (totalNotes - 1 || 1) // 0 to 1
    let value: number
    switch (contour) {
      case "ascending":
        value = t
        break
      case "descending":
        value = 1 - t
        break
      case "arch":
        value = Math.sin(t * Math.PI)
        break
      case "wave":
        value = (Math.sin(t * Math.PI * 2) + 1) / 2
        break
      case "flat":
      default:
        value = 0.5
    }
    contourValues.push(value)
  }

  // Generate notes
  let currentTick = startTick
  const avgTickGap = Math.floor(totalTicks / totalNotes)

  for (let i = 0; i < totalNotes && currentTick < startTick + totalTicks; i++) {
    // Pick pitch based on contour
    const contourIdx = Math.floor(contourValues[i] * (scaleNotes.length - 1))
    let pitch = scaleNotes[contourIdx]

    // Add some randomness
    const randomOffset =
      Math.floor(Math.random() * 3) - 1 // -1, 0, or 1
    const pitchIdx = Math.max(
      0,
      Math.min(scaleNotes.length - 1, contourIdx + randomOffset),
    )
    pitch = scaleNotes[pitchIdx]

    // Pick duration
    const duration = durations[Math.floor(Math.random() * durations.length)]

    // Velocity variation
    const velVar = Math.floor(Math.random() * 20) - 10
    const noteVel = Math.max(60, Math.min(120, velocity + velVar))

    notes.push({
      pitch,
      start: currentTick,
      duration,
      velocity: noteVel,
    })

    // Advance tick with some variation
    const tickVar = Math.floor(Math.random() * 120) - 60
    currentTick += Math.max(120, avgTickGap + tickVar)
  }

  return notes
}

// ============================================================================
// ARPEGGIO GENERATION
// ============================================================================

export function generateArpeggioNotes(
  chordSymbol: string,
  startTick: number,
  duration: number,
  pattern: string,
  rate: number,
  octaves: number,
  velocity: number,
  octave: number = 4,
): Array<{ pitch: number; start: number; duration: number; velocity: number }> {
  const notes: Array<{
    pitch: number
    start: number
    duration: number
    velocity: number
  }> = []

  const chord = parseChord(chordSymbol)
  if (!chord) return notes

  const baseNote = chord.root + (octave + 1) * 12

  // Build the arpeggio note sequence
  let sequence: number[] = []

  // Add notes for each octave
  for (let oct = 0; oct < octaves; oct++) {
    for (const interval of chord.intervals) {
      sequence.push(baseNote + interval + oct * 12)
    }
  }

  // Apply pattern
  switch (pattern) {
    case "down":
      sequence = sequence.reverse()
      break
    case "updown":
      sequence = [...sequence, ...sequence.slice(1, -1).reverse()]
      break
    case "downup":
      sequence = [
        ...sequence.reverse(),
        ...sequence.slice(1, -1).reverse(),
      ].reverse()
      sequence = [...sequence.reverse(), ...sequence.slice(1, -1)]
      break
    case "random":
      sequence = sequence.sort(() => Math.random() - 0.5)
      break
    case "outside_in": {
      const newSeq: number[] = []
      let left = 0
      let right = sequence.length - 1
      while (left <= right) {
        if (left === right) {
          newSeq.push(sequence[left])
        } else {
          newSeq.push(sequence[left], sequence[right])
        }
        left++
        right--
      }
      sequence = newSeq
      break
    }
    case "inside_out": {
      const mid = Math.floor(sequence.length / 2)
      const newSeq: number[] = []
      for (let i = 0; i <= mid; i++) {
        if (mid + i < sequence.length) newSeq.push(sequence[mid + i])
        if (mid - i >= 0 && mid - i !== mid + i)
          newSeq.push(sequence[mid - i])
      }
      sequence = newSeq
      break
    }
    // "up" is default
  }

  // Generate notes
  let tick = startTick
  let idx = 0
  while (tick < startTick + duration) {
    const pitch = sequence[idx % sequence.length]
    notes.push({
      pitch,
      start: tick,
      duration: rate - 30,
      velocity: velocity - (idx % 4) * 3, // Slight accent pattern
    })
    tick += rate
    idx++
  }

  return notes
}

// ============================================================================
// HUMANIZATION
// ============================================================================

export interface HumanizationParams {
  velocityVariation: number
  timingVariation: number
  swing: number
}

export function humanizeNotes(
  notes: Array<{
    id: number
    tick: number
    velocity: number
    duration: number
  }>,
  params: HumanizationParams,
): Array<{ id: number; tick?: number; velocity?: number }> {
  const updates: Array<{ id: number; tick?: number; velocity?: number }> = []

  const eighthNote = 240

  for (const note of notes) {
    const update: { id: number; tick?: number; velocity?: number } = {
      id: note.id,
    }

    // Velocity variation
    if (params.velocityVariation > 0) {
      const velChange =
        Math.floor(Math.random() * params.velocityVariation * 2) -
        params.velocityVariation
      update.velocity = Math.max(
        30,
        Math.min(127, note.velocity + velChange),
      )
    }

    // Timing variation
    let tickChange = 0
    if (params.timingVariation > 0) {
      tickChange =
        Math.floor(Math.random() * params.timingVariation * 2) -
        params.timingVariation
    }

    // Swing (apply to off-beat notes)
    if (params.swing > 0) {
      const posInBeat = note.tick % (eighthNote * 2)
      if (posInBeat >= eighthNote - 30 && posInBeat <= eighthNote + 30) {
        // This is an off-beat note
        const swingOffset = Math.floor((params.swing / 100) * 60)
        tickChange += swingOffset
      }
    }

    if (tickChange !== 0) {
      update.tick = Math.max(0, note.tick + tickChange)
    }

    if (update.velocity !== undefined || update.tick !== undefined) {
      updates.push(update)
    }
  }

  return updates
}

