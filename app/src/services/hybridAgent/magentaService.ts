/**
 * Magenta.js service layer for AI-powered music generation.
 * Provides model loading, caching, and generation functions.
 */

import * as mm from "@magenta/music"

// ============================================================================
// LOGGING UTILITIES
// ============================================================================

const LOG_PREFIX = "[Magenta]"
const LOG_STYLES = {
  info: "color: #4CAF50; font-weight: bold",
  warn: "color: #FF9800; font-weight: bold",
  error: "color: #F44336; font-weight: bold",
  model: "color: #2196F3; font-weight: bold",
  generate: "color: #9C27B0; font-weight: bold",
}

function logInfo(message: string, ...args: unknown[]) {
  console.log(`%c${LOG_PREFIX} ${message}`, LOG_STYLES.info, ...args)
}

function logModel(message: string, ...args: unknown[]) {
  console.log(`%c${LOG_PREFIX} [MODEL] ${message}`, LOG_STYLES.model, ...args)
}

function logGenerate(message: string, ...args: unknown[]) {
  console.log(`%c${LOG_PREFIX} [GENERATE] ${message}`, LOG_STYLES.generate, ...args)
}

function logWarn(message: string, ...args: unknown[]) {
  console.warn(`%c${LOG_PREFIX} ${message}`, LOG_STYLES.warn, ...args)
}

function logError(message: string, ...args: unknown[]) {
  console.error(`%c${LOG_PREFIX} ${message}`, LOG_STYLES.error, ...args)
}

// ============================================================================
// MODEL CHECKPOINTS (hosted by Google)
// ============================================================================

const CHECKPOINTS = {
  // Drum models
  drumsRnn:
    "https://storage.googleapis.com/magentadata/js/checkpoints/music_rnn/drum_kit_rnn",

  // Melody models
  melodyRnn:
    "https://storage.googleapis.com/magentadata/js/checkpoints/music_rnn/basic_rnn",
  melodyRnnAttention:
    "https://storage.googleapis.com/magentadata/js/checkpoints/music_rnn/attention_rnn",

  // Improv (melody over chords)
  improvRnn:
    "https://storage.googleapis.com/magentadata/js/checkpoints/music_rnn/chord_pitches_improv",

  // MusicVAE models (more powerful, larger)
  musicVaeDrums2Bar:
    "https://storage.googleapis.com/magentadata/js/checkpoints/music_vae/drums_2bar_lokl_small",
  musicVaeDrums4Bar:
    "https://storage.googleapis.com/magentadata/js/checkpoints/music_vae/drums_4bar_med_lokl",
  musicVaeMel2Bar:
    "https://storage.googleapis.com/magentadata/js/checkpoints/music_vae/mel_2bar_small",
  musicVaeMel4Bar:
    "https://storage.googleapis.com/magentadata/js/checkpoints/music_vae/mel_4bar_med_lokl",
  musicVaeTrioSmall:
    "https://storage.googleapis.com/magentadata/js/checkpoints/music_vae/trio_4bar",
}

// ============================================================================
// MODEL CACHE
// ============================================================================

interface ModelCache {
  drumsRnn?: mm.MusicRNN
  melodyRnn?: mm.MusicRNN
  improvRnn?: mm.MusicRNN
  musicVaeDrums?: mm.MusicVAE
  musicVaeMelody?: mm.MusicVAE
}

const modelCache: ModelCache = {}
const loadingPromises: Map<string, Promise<unknown>> = new Map()

/**
 * Get or load a model with caching.
 */
async function getModel<T>(
  key: keyof ModelCache,
  loader: () => Promise<T>,
): Promise<T> {
  if (modelCache[key]) {
    return modelCache[key] as T
  }

  // Check if already loading
  if (loadingPromises.has(key)) {
    return loadingPromises.get(key) as Promise<T>
  }

  // Start loading
  const loadPromise = loader().then((model) => {
    ;(modelCache as Record<string, unknown>)[key] = model
    loadingPromises.delete(key)
    return model
  })

  loadingPromises.set(key, loadPromise)
  return loadPromise
}

// ============================================================================
// MODEL LOADERS
// ============================================================================

export async function getDrumsRnn(): Promise<mm.MusicRNN> {
  return getModel("drumsRnn", async () => {
    logModel("Loading DrumsRNN from:", CHECKPOINTS.drumsRnn)
    const startTime = performance.now()
    const model = new mm.MusicRNN(CHECKPOINTS.drumsRnn)
    await model.initialize()
    const loadTime = ((performance.now() - startTime) / 1000).toFixed(2)
    logModel(`DrumsRNN loaded successfully (${loadTime}s)`)
    return model
  })
}

export async function getMelodyRnn(): Promise<mm.MusicRNN> {
  return getModel("melodyRnn", async () => {
    logModel("Loading MelodyRNN (attention) from:", CHECKPOINTS.melodyRnnAttention)
    const startTime = performance.now()
    const model = new mm.MusicRNN(CHECKPOINTS.melodyRnnAttention)
    await model.initialize()
    const loadTime = ((performance.now() - startTime) / 1000).toFixed(2)
    logModel(`MelodyRNN loaded successfully (${loadTime}s)`)
    return model
  })
}

export async function getImprovRnn(): Promise<mm.MusicRNN> {
  return getModel("improvRnn", async () => {
    logModel("Loading ImprovRNN (chord-following) from:", CHECKPOINTS.improvRnn)
    const startTime = performance.now()
    const model = new mm.MusicRNN(CHECKPOINTS.improvRnn)
    await model.initialize()
    const loadTime = ((performance.now() - startTime) / 1000).toFixed(2)
    logModel(`ImprovRNN loaded successfully (${loadTime}s)`)
    return model
  })
}

export async function getMusicVaeDrums(): Promise<mm.MusicVAE> {
  return getModel("musicVaeDrums", async () => {
    logModel("Loading MusicVAE (drums 4-bar) from:", CHECKPOINTS.musicVaeDrums4Bar)
    const startTime = performance.now()
    const model = new mm.MusicVAE(CHECKPOINTS.musicVaeDrums4Bar)
    await model.initialize()
    const loadTime = ((performance.now() - startTime) / 1000).toFixed(2)
    logModel(`MusicVAE (drums) loaded successfully (${loadTime}s)`)
    return model
  })
}

export async function getMusicVaeMelody(): Promise<mm.MusicVAE> {
  return getModel("musicVaeMelody", async () => {
    logModel("Loading MusicVAE (melody 4-bar) from:", CHECKPOINTS.musicVaeMel4Bar)
    const startTime = performance.now()
    const model = new mm.MusicVAE(CHECKPOINTS.musicVaeMel4Bar)
    await model.initialize()
    const loadTime = ((performance.now() - startTime) / 1000).toFixed(2)
    logModel(`MusicVAE (melody) loaded successfully (${loadTime}s)`)
    return model
  })
}

// ============================================================================
// NOTE SEQUENCE CONVERSION
// ============================================================================

export interface AppNote {
  pitch: number
  start: number // ticks
  duration: number // ticks
  velocity: number
}

const TICKS_PER_QUARTER = 480
const MAGENTA_STEPS_PER_QUARTER = 4 // Magenta uses 16th note quantization

/**
 * Convert app ticks to Magenta quantized steps.
 */
function ticksToSteps(ticks: number): number {
  return Math.round((ticks / TICKS_PER_QUARTER) * MAGENTA_STEPS_PER_QUARTER)
}

/**
 * Convert Magenta quantized steps to app ticks.
 */
function stepsToTicks(steps: number): number {
  return Math.round((steps / MAGENTA_STEPS_PER_QUARTER) * TICKS_PER_QUARTER)
}

/**
 * Convert app notes to Magenta NoteSequence.
 */
export function notesToNoteSequence(
  notes: AppNote[],
  totalTicks: number,
): mm.INoteSequence {
  const totalSteps = ticksToSteps(totalTicks)

  return {
    notes: notes.map((n) => ({
      pitch: n.pitch,
      quantizedStartStep: ticksToSteps(n.start),
      quantizedEndStep: ticksToSteps(n.start + n.duration),
      velocity: n.velocity,
    })),
    totalQuantizedSteps: totalSteps,
    quantizationInfo: { stepsPerQuarter: MAGENTA_STEPS_PER_QUARTER },
  }
}

/**
 * Convert Magenta NoteSequence to app notes.
 */
export function noteSequenceToNotes(
  seq: mm.INoteSequence,
  startTickOffset: number = 0,
): AppNote[] {
  if (!seq.notes) return []

  return seq.notes.map((n) => ({
    pitch: n.pitch ?? 60,
    start: stepsToTicks(n.quantizedStartStep ?? 0) + startTickOffset,
    duration: Math.max(
      stepsToTicks((n.quantizedEndStep ?? 1) - (n.quantizedStartStep ?? 0)),
      120, // Minimum duration of 1/16 note
    ),
    velocity: n.velocity ?? 80,
  }))
}

// ============================================================================
// DRUM PATTERN GENERATION
// ============================================================================

// GM Drum map for seeds - comprehensive style patterns
const DRUM_SEED_NOTES: Record<string, Array<{ pitch: number; step: number; velocity?: number }>> = {
  // Standard rock - driving 8th note feel
  rock: [
    { pitch: 36, step: 0, velocity: 100 }, // Kick on 1
    { pitch: 42, step: 0 }, // HH
    { pitch: 42, step: 2 },
    { pitch: 38, step: 4, velocity: 100 }, // Snare on 2
    { pitch: 42, step: 4 },
    { pitch: 42, step: 6 },
    { pitch: 36, step: 8 }, // Kick on 3
    { pitch: 42, step: 8 },
    { pitch: 42, step: 10 },
    { pitch: 38, step: 12, velocity: 100 }, // Snare on 4
    { pitch: 42, step: 12 },
    { pitch: 42, step: 14 },
  ],
  // Pop-punk / Blink-182 style - fast, driving
  punk: [
    { pitch: 36, step: 0, velocity: 110 }, // Kick
    { pitch: 42, step: 0 },
    { pitch: 42, step: 1 },
    { pitch: 42, step: 2 },
    { pitch: 42, step: 3 },
    { pitch: 38, step: 4, velocity: 110 }, // Snare
    { pitch: 42, step: 4 },
    { pitch: 42, step: 5 },
    { pitch: 42, step: 6 },
    { pitch: 36, step: 7 },
    { pitch: 36, step: 8, velocity: 100 },
    { pitch: 42, step: 8 },
    { pitch: 42, step: 9 },
    { pitch: 42, step: 10 },
    { pitch: 42, step: 11 },
    { pitch: 38, step: 12, velocity: 110 },
    { pitch: 42, step: 12 },
    { pitch: 42, step: 13 },
    { pitch: 42, step: 14 },
    { pitch: 36, step: 15 },
  ],
  // Metal - double kick, aggressive
  metal: [
    { pitch: 36, step: 0, velocity: 120 },
    { pitch: 49, step: 0, velocity: 100 }, // Crash
    { pitch: 36, step: 2 },
    { pitch: 38, step: 4, velocity: 120 },
    { pitch: 36, step: 4 },
    { pitch: 36, step: 6 },
    { pitch: 36, step: 8, velocity: 110 },
    { pitch: 42, step: 8 },
    { pitch: 36, step: 10 },
    { pitch: 38, step: 12, velocity: 120 },
    { pitch: 36, step: 12 },
    { pitch: 36, step: 14 },
  ],
  // Jazz - swing feel with ride
  jazz: [
    { pitch: 51, step: 0, velocity: 70 }, // Ride
    { pitch: 51, step: 3 }, // Swung 8th
    { pitch: 51, step: 4 },
    { pitch: 51, step: 7 },
    { pitch: 51, step: 8 },
    { pitch: 51, step: 11 },
    { pitch: 51, step: 12 },
    { pitch: 51, step: 15 },
    { pitch: 44, step: 4, velocity: 60 }, // Hi-hat pedal on 2
    { pitch: 44, step: 12, velocity: 60 }, // Hi-hat pedal on 4
  ],
  // Funk - syncopated, ghost notes
  funk: [
    { pitch: 36, step: 0, velocity: 100 }, // Kick
    { pitch: 42, step: 0 },
    { pitch: 42, step: 2 },
    { pitch: 38, step: 4, velocity: 100 }, // Snare
    { pitch: 42, step: 4 },
    { pitch: 38, step: 6, velocity: 50 }, // Ghost snare
    { pitch: 36, step: 7 },
    { pitch: 42, step: 8 },
    { pitch: 38, step: 10, velocity: 50 }, // Ghost
    { pitch: 42, step: 10 },
    { pitch: 38, step: 12, velocity: 100 },
    { pitch: 42, step: 12 },
    { pitch: 42, step: 14 },
    { pitch: 36, step: 15 },
  ],
  // Hip-hop / trap style
  hiphop: [
    { pitch: 36, step: 0, velocity: 110 },
    { pitch: 42, step: 2 },
    { pitch: 38, step: 4, velocity: 100 },
    { pitch: 42, step: 6 },
    { pitch: 42, step: 8 },
    { pitch: 36, step: 10 },
    { pitch: 38, step: 12, velocity: 100 },
    { pitch: 42, step: 14 },
  ],
  // Electronic / EDM - four on the floor
  electronic: [
    { pitch: 36, step: 0, velocity: 120 }, // Kick on every beat
    { pitch: 42, step: 2 },
    { pitch: 36, step: 4, velocity: 120 },
    { pitch: 46, step: 4 }, // Open HH
    { pitch: 42, step: 6 },
    { pitch: 36, step: 8, velocity: 120 },
    { pitch: 42, step: 10 },
    { pitch: 36, step: 12, velocity: 120 },
    { pitch: 46, step: 12 },
    { pitch: 42, step: 14 },
  ],
  // Latin / Bossa
  latin: [
    { pitch: 36, step: 0 },
    { pitch: 37, step: 3, velocity: 70 }, // Side stick
    { pitch: 36, step: 6 },
    { pitch: 37, step: 7, velocity: 70 },
    { pitch: 37, step: 10, velocity: 70 },
    { pitch: 36, step: 12 },
    { pitch: 37, step: 15, velocity: 70 },
  ],
  // Ballad - sparse, emotional
  ballad: [
    { pitch: 36, step: 0, velocity: 70 },
    { pitch: 42, step: 4, velocity: 60 },
    { pitch: 38, step: 8, velocity: 75 },
    { pitch: 42, step: 12, velocity: 60 },
  ],
  // Country - train beat
  country: [
    { pitch: 36, step: 0 },
    { pitch: 42, step: 0 },
    { pitch: 37, step: 2 }, // Side stick
    { pitch: 42, step: 2 },
    { pitch: 36, step: 4 },
    { pitch: 42, step: 4 },
    { pitch: 37, step: 6 },
    { pitch: 42, step: 6 },
    { pitch: 36, step: 8 },
    { pitch: 42, step: 8 },
    { pitch: 37, step: 10 },
    { pitch: 42, step: 10 },
    { pitch: 36, step: 12 },
    { pitch: 42, step: 12 },
    { pitch: 37, step: 14 },
    { pitch: 42, step: 14 },
  ],
}

// Map style keywords to drum patterns
function getDrumStyleKey(style: string): string {
  const s = style.toLowerCase()
  if (s.includes("punk") || s.includes("blink") || s.includes("pop-punk") || s.includes("pop punk")) return "punk"
  if (s.includes("metal") || s.includes("heavy")) return "metal"
  if (s.includes("jazz") || s.includes("swing")) return "jazz"
  if (s.includes("funk") || s.includes("groove")) return "funk"
  if (s.includes("hip") || s.includes("hop") || s.includes("trap") || s.includes("rap")) return "hiphop"
  if (s.includes("electro") || s.includes("edm") || s.includes("house") || s.includes("techno")) return "electronic"
  if (s.includes("latin") || s.includes("bossa") || s.includes("samba")) return "latin"
  if (s.includes("ballad") || s.includes("slow") || s.includes("soft")) return "ballad"
  if (s.includes("country") || s.includes("folk")) return "country"
  if (s.includes("rock")) return "rock"
  return "rock" // Default
}

/**
 * Create a seed sequence for drum generation based on style.
 */
function createDrumSeed(style: string): mm.INoteSequence {
  const styleKey = getDrumStyleKey(style)
  const seedNotes = DRUM_SEED_NOTES[styleKey] ?? DRUM_SEED_NOTES.rock

  logGenerate(`Using drum seed pattern: ${styleKey} (${seedNotes.length} hits)`)

  return {
    notes: seedNotes.map((n) => ({
      pitch: n.pitch,
      quantizedStartStep: n.step,
      quantizedEndStep: n.step + 1,
      velocity: n.velocity ?? (80 + Math.floor(Math.random() * 20)),
      isDrum: true,
    })),
    totalQuantizedSteps: 16,
    quantizationInfo: { stepsPerQuarter: MAGENTA_STEPS_PER_QUARTER },
  }
}

export interface DrumGenerationOptions {
  style: string
  bars: number
  temperature: number
  startTick: number
}

/**
 * Generate drum pattern using DrumsRNN.
 */
export async function generateDrumPatternMagenta(
  options: DrumGenerationOptions,
): Promise<AppNote[]> {
  const { style, bars, temperature, startTick } = options

  logGenerate("=== DRUM PATTERN (DrumsRNN) ===")
  logGenerate(`Style: ${style}, Bars: ${bars}, Temperature: ${temperature}, StartTick: ${startTick}`)

  const model = await getDrumsRnn()
  const seed = createDrumSeed(style)
  const stepsToGenerate = bars * 16 // 16 steps per bar (16th notes)

  logGenerate(`Seed created with ${seed.notes?.length ?? 0} notes, generating ${stepsToGenerate} steps...`)

  const startTime = performance.now()
  try {
    const sequence = await model.continueSequence(
      seed,
      stepsToGenerate,
      temperature,
    )
    const genTime = ((performance.now() - startTime) / 1000).toFixed(2)

    logGenerate(`Raw DrumsRNN output: ${sequence.notes?.length ?? 0} notes, ${sequence.totalQuantizedSteps} steps`)

    if (!sequence.notes || sequence.notes.length === 0) {
      logWarn(`DrumsRNN returned 0 notes!`)
      return []
    }

    const notes = noteSequenceToNotes(sequence, startTick)
    logGenerate(`Generated ${notes.length} drum notes in ${genTime}s`)
    logGenerate("Sample notes:", notes.slice(0, 5).map(n => `pitch:${n.pitch}@${n.start}`))

    return notes
  } catch (error) {
    logError(`DrumsRNN generation failed: ${error}`)
    return []
  }
}

/**
 * Generate drum pattern using MusicVAE (more varied).
 */
export async function generateDrumPatternVAE(
  options: DrumGenerationOptions,
): Promise<AppNote[]> {
  const { bars, temperature, startTick, style } = options

  logGenerate("=== DRUM PATTERN (MusicVAE) ===")
  logGenerate(`Style hint: ${style}, Bars: ${bars}, Temperature: ${temperature}, StartTick: ${startTick}`)

  try {
    const model = await getMusicVaeDrums()
    const numSamples = Math.ceil(bars / 4) // VAE generates 4 bars at a time

    logGenerate(`Sampling ${numSamples} x 4-bar patterns from latent space...`)

    const startTime = performance.now()
    const sequences = await model.sample(numSamples, temperature)
    const genTime = ((performance.now() - startTime) / 1000).toFixed(2)

    logGenerate(`MusicVAE returned ${sequences.length} sequences`)

    const allNotes: AppNote[] = []

    sequences.forEach((seq, i) => {
      logGenerate(`Sequence ${i + 1}: ${seq.notes?.length ?? 0} raw notes`)
      const offset = startTick + i * 4 * TICKS_PER_QUARTER * 4 // 4 bars offset
      const notes = noteSequenceToNotes(seq, offset)
      logGenerate(`Sample ${i + 1}: ${notes.length} converted notes`)
      allNotes.push(...notes)
    })

    // Trim to requested bars
    const maxTick = startTick + bars * TICKS_PER_QUARTER * 4
    const trimmedNotes = allNotes.filter((n) => n.start < maxTick)

    logGenerate(`Generated ${trimmedNotes.length} drum notes total in ${genTime}s (VAE)`)

    return trimmedNotes
  } catch (error) {
    logError(`MusicVAE drums generation failed: ${error}`)
    return []
  }
}

// ============================================================================
// MELODY GENERATION
// ============================================================================

export interface MelodyGenerationOptions {
  scale: string
  bars: number
  temperature: number
  startTick: number
  rangeLow: number
  rangeHigh: number
  style?: string // Genre hint for seed generation
}

// Style-specific melody seed patterns (scale degrees and rhythmic patterns)
// Format: { degrees: scale degrees to use, rhythm: step positions, velocities: optional velocity curve }
const MELODY_SEED_PATTERNS: Record<string, {
  degrees: number[]
  steps: number[]
  durations: number[]
  velocities?: number[]
}> = {
  // Rock - strong, rhythmic, power chord feel
  rock: {
    degrees: [0, 0, 4, 4, 5, 5, 4, -1], // Power chord movement
    steps: [0, 2, 4, 6, 8, 10, 12, 14],
    durations: [2, 2, 2, 2, 2, 2, 2, 2],
    velocities: [100, 80, 100, 80, 100, 80, 100, 70],
  },
  // Punk - fast, aggressive, simple
  punk: {
    degrees: [0, 2, 4, 5, 4, 2, 0, 0],
    steps: [0, 1, 2, 3, 4, 5, 6, 7],
    durations: [1, 1, 1, 1, 1, 1, 1, 1],
    velocities: [110, 100, 110, 100, 110, 100, 110, 100],
  },
  // Metal - dark, aggressive
  metal: {
    degrees: [0, 0, -1, 0, 3, 0, -1, -2],
    steps: [0, 2, 4, 6, 8, 10, 12, 14],
    durations: [2, 2, 2, 2, 2, 2, 2, 2],
    velocities: [120, 110, 120, 110, 120, 110, 120, 110],
  },
  // Jazz - chromatic, complex rhythm
  jazz: {
    degrees: [0, 2, 4, 6, 5, 4, 3, 2],
    steps: [0, 3, 4, 7, 8, 10, 12, 15],
    durations: [3, 1, 3, 1, 2, 2, 3, 1],
    velocities: [70, 60, 75, 65, 70, 60, 75, 60],
  },
  // Pop - catchy, melodic
  pop: {
    degrees: [0, 2, 4, 2, 0, 4, 5, 4],
    steps: [0, 2, 4, 6, 8, 10, 12, 14],
    durations: [2, 2, 2, 2, 2, 2, 2, 2],
    velocities: [90, 80, 95, 80, 90, 85, 95, 80],
  },
  // Ballad - slow, expressive
  ballad: {
    degrees: [0, 2, 4, 5, 4, 2, 0, -1],
    steps: [0, 4, 8, 10, 12, 14, 16, 20],
    durations: [4, 4, 2, 2, 2, 2, 4, 4],
    velocities: [70, 75, 80, 85, 80, 75, 70, 65],
  },
  // Electronic - repetitive, arpeggiated
  electronic: {
    degrees: [0, 4, 7, 4, 0, 4, 7, 11],
    steps: [0, 2, 4, 6, 8, 10, 12, 14],
    durations: [2, 2, 2, 2, 2, 2, 2, 2],
    velocities: [100, 90, 100, 90, 100, 90, 100, 95],
  },
  // Funk - syncopated, groovy
  funk: {
    degrees: [0, -1, 0, 3, 4, -1, 4, 3],
    steps: [0, 3, 4, 6, 8, 11, 12, 14],
    durations: [3, 1, 2, 2, 3, 1, 2, 2],
    velocities: [100, 70, 90, 80, 100, 70, 90, 80],
  },
  // Hip-hop - sparse, rhythmic
  hiphop: {
    degrees: [0, 0, 3, 0, 0, 3, 5, 3],
    steps: [0, 4, 6, 8, 12, 14, 16, 18],
    durations: [4, 2, 2, 4, 2, 2, 2, 6],
    velocities: [100, 80, 90, 100, 80, 90, 95, 85],
  },
  // Latin - rhythmic, syncopated
  latin: {
    degrees: [0, 2, 4, 5, 4, 2, 3, 2],
    steps: [0, 3, 4, 6, 8, 11, 12, 14],
    durations: [3, 1, 2, 2, 3, 1, 2, 2],
    velocities: [90, 70, 85, 80, 90, 70, 85, 75],
  },
  // Country - twangy, melodic
  country: {
    degrees: [0, 2, 4, 5, 7, 5, 4, 2],
    steps: [0, 2, 4, 6, 8, 10, 12, 14],
    durations: [2, 2, 2, 2, 2, 2, 2, 2],
    velocities: [85, 75, 90, 80, 95, 80, 85, 75],
  },
}

// Map style keywords to melody patterns
function getMelodyStyleKey(style?: string): string {
  if (!style) return "rock"
  const s = style.toLowerCase()
  if (s.includes("punk") || s.includes("blink") || s.includes("pop-punk") || s.includes("pop punk")) return "punk"
  if (s.includes("metal") || s.includes("heavy")) return "metal"
  if (s.includes("jazz") || s.includes("swing")) return "jazz"
  if (s.includes("funk") || s.includes("groove")) return "funk"
  if (s.includes("hip") || s.includes("hop") || s.includes("trap") || s.includes("rap")) return "hiphop"
  if (s.includes("electro") || s.includes("edm") || s.includes("house") || s.includes("techno")) return "electronic"
  if (s.includes("latin") || s.includes("bossa") || s.includes("samba")) return "latin"
  if (s.includes("ballad") || s.includes("slow") || s.includes("soft")) return "ballad"
  if (s.includes("country") || s.includes("folk")) return "country"
  if (s.includes("pop")) return "pop"
  if (s.includes("rock")) return "rock"
  return "rock" // Default
}

const NOTE_MAP: Record<string, number> = {
  C: 0, "C#": 1, Db: 1, D: 2, "D#": 3, Eb: 3, E: 4, F: 5,
  "F#": 6, Gb: 6, G: 7, "G#": 8, Ab: 8, A: 9, "A#": 10, Bb: 10, B: 11,
}

/**
 * Create a seed melody based on scale and style.
 */
function createMelodySeed(
  scale: string,
  rangeLow: number,
  style?: string,
): mm.INoteSequence {
  // Extract root from scale name (e.g., "C major" -> C, "Am" -> A)
  const rootMatch = scale.match(/^([A-G][#b]?)/)
  const rootName = rootMatch ? rootMatch[1] : "C"

  const rootPitch = NOTE_MAP[rootName] ?? 0
  const basePitch = Math.max(rangeLow, 60 + rootPitch - 12)

  // Determine scale intervals
  const isMinor = scale.toLowerCase().includes("minor") ||
    (scale.toLowerCase().includes("m") && !scale.toLowerCase().includes("maj"))
  const intervals = isMinor
    ? [0, 2, 3, 5, 7, 8, 10] // Natural minor
    : [0, 2, 4, 5, 7, 9, 11] // Major

  // Get style-specific pattern
  const styleKey = getMelodyStyleKey(style)
  const pattern = MELODY_SEED_PATTERNS[styleKey] ?? MELODY_SEED_PATTERNS.rock

  logGenerate(`Using melody seed pattern: ${styleKey}`)

  // Convert scale degrees to pitches
  const seedNotes = pattern.degrees.map((degree, i) => {
    // Handle negative degrees (going below root)
    let pitch: number
    if (degree < 0) {
      // Go down from root by the interval
      const absDegree = Math.abs(degree)
      const intervalFromOctaveBelow = intervals[intervals.length - absDegree] ?? 0
      pitch = basePitch - (12 - intervalFromOctaveBelow)
    } else {
      // Normal scale degree
      const octave = Math.floor(degree / 7)
      const degreeInOctave = degree % 7
      pitch = basePitch + octave * 12 + (intervals[degreeInOctave] ?? 0)
    }

    return {
      pitch,
      quantizedStartStep: pattern.steps[i] ?? i * 2,
      quantizedEndStep: (pattern.steps[i] ?? i * 2) + (pattern.durations[i] ?? 2),
      velocity: pattern.velocities?.[i] ?? 80,
    }
  })

  const totalSteps = Math.max(...seedNotes.map(n => n.quantizedEndStep), 16)

  return {
    notes: seedNotes,
    totalQuantizedSteps: totalSteps,
    quantizationInfo: { stepsPerQuarter: MAGENTA_STEPS_PER_QUARTER },
  }
}

/**
 * Generate melody using MelodyRNN.
 */
export async function generateMelodyMagenta(
  options: MelodyGenerationOptions,
): Promise<AppNote[]> {
  const { scale, bars, temperature, startTick, rangeLow, rangeHigh, style } = options

  logGenerate("=== MELODY (MelodyRNN) ===")
  logGenerate(`Scale: ${scale}, Style: ${style ?? 'default'}, Bars: ${bars}, Temperature: ${temperature}`)
  logGenerate(`Range: ${rangeLow}-${rangeHigh}, StartTick: ${startTick}`)

  const model = await getMelodyRnn()
  const seed = createMelodySeed(scale, rangeLow, style)
  const stepsToGenerate = bars * 16

  logGenerate(`Seed created with ${seed.notes?.length ?? 0} notes, generating ${stepsToGenerate} steps...`)
  logGenerate(`Seed notes: ${JSON.stringify(seed.notes?.slice(0, 3))}`)

  const startTime = performance.now()
  try {
    const sequence = await model.continueSequence(
      seed,
      stepsToGenerate,
      temperature,
    )
    const genTime = ((performance.now() - startTime) / 1000).toFixed(2)

    logGenerate(`Raw MelodyRNN output: ${sequence.notes?.length ?? 0} notes, ${sequence.totalQuantizedSteps} steps`)
    
    if (!sequence.notes || sequence.notes.length === 0) {
      logWarn(`MelodyRNN returned 0 notes!`)
      return []
    }

    // Filter notes to requested range
    let notes = noteSequenceToNotes(sequence, startTick)
    const beforeFilter = notes.length
    notes = notes.filter((n) => n.pitch >= rangeLow && n.pitch <= rangeHigh)

    logGenerate(`Generated ${notes.length} melody notes in ${genTime}s (${beforeFilter - notes.length} filtered out of range)`)
    logGenerate("Sample notes:", notes.slice(0, 5).map(n => `pitch:${n.pitch}@${n.start}`))

    return notes
  } catch (error) {
    logError(`MelodyRNN generation failed: ${error}`)
    return []
  }
}

/**
 * Generate melody using MusicVAE (more varied).
 */
export async function generateMelodyVAE(
  options: MelodyGenerationOptions,
): Promise<AppNote[]> {
  const { bars, temperature, startTick, rangeLow, rangeHigh, scale, style } = options

  logGenerate("=== MELODY (MusicVAE) ===")
  logGenerate(`Scale hint: ${scale}, Style: ${style ?? 'default'}, Bars: ${bars}, Temperature: ${temperature}`)
  logGenerate(`Range: ${rangeLow}-${rangeHigh}, StartTick: ${startTick}`)

  try {
    const model = await getMusicVaeMelody()
    const numSamples = Math.ceil(bars / 4)

    logGenerate(`Sampling ${numSamples} x 4-bar patterns from latent space...`)

    const startTime = performance.now()
    const sequences = await model.sample(numSamples, temperature)
    const genTime = ((performance.now() - startTime) / 1000).toFixed(2)

    logGenerate(`MusicVAE returned ${sequences.length} sequences`)

    const allNotes: AppNote[] = []

    sequences.forEach((seq, i) => {
      logGenerate(`Sequence ${i + 1}: ${seq.notes?.length ?? 0} raw notes`)
      const offset = startTick + i * 4 * TICKS_PER_QUARTER * 4
      const notes = noteSequenceToNotes(seq, offset)
      logGenerate(`Sample ${i + 1}: ${notes.length} converted notes`)
      allNotes.push(...notes)
    })

    // Trim and filter
    const maxTick = startTick + bars * TICKS_PER_QUARTER * 4
    const trimmedNotes = allNotes
      .filter((n) => n.start < maxTick)
      .filter((n) => n.pitch >= rangeLow && n.pitch <= rangeHigh)

    logGenerate(`Generated ${trimmedNotes.length} melody notes total in ${genTime}s (VAE)`)

    return trimmedNotes
  } catch (error) {
    logError(`MusicVAE melody generation failed: ${error}`)
    return []
  }
}

// ============================================================================
// IMPROV (MELODY OVER CHORDS)
// ============================================================================

export interface ImprovGenerationOptions {
  chordProgression: string[] // e.g., ["C", "Am", "F", "G"]
  bars: number
  temperature: number
  startTick: number
  ticksPerChord: number
}

/**
 * Generate melody that follows chord changes using ImprovRNN.
 */
export async function generateImprovMagenta(
  options: ImprovGenerationOptions,
): Promise<AppNote[]> {
  const { chordProgression, bars, temperature, startTick, ticksPerChord } = options

  logGenerate("=== IMPROV MELODY (ImprovRNN) ===")
  logGenerate(`Chords: ${chordProgression.join(" → ")}`)
  logGenerate(`Bars: ${bars}, Temperature: ${temperature}, TicksPerChord: ${ticksPerChord}`)

  const model = await getImprovRnn()

  // ImprovRNN needs one chord per step
  // 4 steps per beat, 4 beats per bar
  const stepsPerBar = 16
  const totalSteps = bars * stepsPerBar
  const stepsPerChord = Math.floor(totalSteps / chordProgression.length)
  
  // Build chord sequence: one chord symbol per step
  const chordSequence: string[] = []
  for (const chord of chordProgression) {
    for (let i = 0; i < stepsPerChord; i++) {
      chordSequence.push(chord)
    }
  }
  // Pad to exact length if needed
  while (chordSequence.length < totalSteps) {
    chordSequence.push(chordProgression[chordProgression.length - 1])
  }

  logGenerate(`Chord sequence: ${stepsPerChord} steps per chord, ${chordSequence.length} total steps`)

  // Create a proper seed with the model's expected structure
  const seed: mm.INoteSequence = {
    notes: [
      { pitch: 60, quantizedStartStep: 0, quantizedEndStep: 2, velocity: 80 },
      { pitch: 62, quantizedStartStep: 2, quantizedEndStep: 4, velocity: 80 },
    ],
    totalQuantizedSteps: 4,
    quantizationInfo: { stepsPerQuarter: 4 },
  }

  logGenerate(`Generating ${totalSteps} steps of melody over ${chordSequence.length} chord steps...`)

  const startTime = performance.now()
  
  try {
    // Log the full input for debugging
    logGenerate(`Seed notes: ${seed.notes?.length}, chords sample: ${chordSequence.slice(0, 5).join(', ')}...`)
    
    const sequence = await model.continueSequence(
      seed,
      totalSteps,
      temperature,
      chordSequence,
    )
    const genTime = ((performance.now() - startTime) / 1000).toFixed(2)

    // Debug: log the raw sequence completely
    logGenerate(`Raw sequence totalQuantizedSteps: ${sequence.totalQuantizedSteps}`)
    logGenerate(`Raw sequence notes: ${JSON.stringify(sequence.notes?.slice(0, 3))}`)
    
    if (sequence.notes && sequence.notes.length > 0) {
      logGenerate(`First note: pitch=${sequence.notes[0].pitch}, start=${sequence.notes[0].quantizedStartStep}, end=${sequence.notes[0].quantizedEndStep}`)
      const notes = noteSequenceToNotes(sequence, startTick)
      logGenerate(`Generated ${notes.length} improv notes in ${genTime}s`)
      return notes
    }
    
    // ImprovRNN returned no notes - try basic MelodyRNN instead
    logWarn(`ImprovRNN returned 0 notes, trying MelodyRNN...`)
    return generateMelodyMagenta({
      scale: "C major",
      bars,
      temperature,
      startTick,
      rangeLow: 48,
      rangeHigh: 84,
    })
  } catch (error) {
    logError(`ImprovRNN generation failed: ${error}`)
    // Fallback to basic melody generation
    logGenerate("Falling back to basic melody RNN...")
    return generateMelodyMagenta({
      scale: "C major",
      bars,
      temperature,
      startTick,
      rangeLow: 48,
      rangeHigh: 84,
    })
  }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Check if Magenta models are available (for graceful fallback).
 */
export function isMagentaAvailable(): boolean {
  const available = typeof mm !== "undefined"
  logInfo(`Magenta availability check: ${available ? "✅ AVAILABLE" : "❌ NOT AVAILABLE"}`)
  return available
}

/**
 * Preload commonly used models.
 */
export async function preloadModels(): Promise<void> {
  logInfo("Preloading commonly used models...")
  const startTime = performance.now()
  await Promise.all([getDrumsRnn(), getMelodyRnn()])
  const loadTime = ((performance.now() - startTime) / 1000).toFixed(2)
  logInfo(`Models preloaded in ${loadTime}s`)
}

/**
 * Clear model cache to free memory.
 */
export function clearModelCache(): void {
  const cachedModels = Object.keys(modelCache).filter(k => modelCache[k as keyof ModelCache])
  logInfo(`Clearing ${cachedModels.length} cached models:`, cachedModels)
  
  Object.keys(modelCache).forEach((key) => {
    const model = modelCache[key as keyof ModelCache]
    if (model && "dispose" in model) {
      (model as { dispose: () => void }).dispose()
    }
  })
  Object.keys(modelCache).forEach(
    (key) => delete modelCache[key as keyof ModelCache],
  )
  logInfo("Model cache cleared")
}

/**
 * Get current cache status for debugging.
 */
export function getModelCacheStatus(): Record<string, boolean> {
  const status: Record<string, boolean> = {}
  Object.keys(modelCache).forEach((key) => {
    status[key] = !!modelCache[key as keyof ModelCache]
  })
  return status
}

