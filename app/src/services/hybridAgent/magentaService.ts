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
    modelCache[key] = model as ModelCache[keyof ModelCache]
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

// GM Drum map for seeds
const DRUM_SEED_NOTES = {
  rock: [
    { pitch: 36, step: 0 }, // Kick on 1
    { pitch: 42, step: 0 }, // HH
    { pitch: 38, step: 4 }, // Snare on 2
    { pitch: 42, step: 4 },
    { pitch: 36, step: 8 }, // Kick on 3
    { pitch: 42, step: 8 },
    { pitch: 38, step: 12 }, // Snare on 4
    { pitch: 42, step: 12 },
  ],
  jazz: [
    { pitch: 51, step: 0 }, // Ride
    { pitch: 51, step: 3 }, // Swung
    { pitch: 51, step: 4 },
    { pitch: 51, step: 7 },
    { pitch: 51, step: 8 },
    { pitch: 51, step: 11 },
    { pitch: 51, step: 12 },
    { pitch: 51, step: 15 },
  ],
  funk: [
    { pitch: 36, step: 0 }, // Kick
    { pitch: 42, step: 2 },
    { pitch: 38, step: 4 }, // Snare
    { pitch: 42, step: 6 },
    { pitch: 36, step: 7 },
    { pitch: 42, step: 10 },
    { pitch: 38, step: 12 },
    { pitch: 42, step: 14 },
  ],
  hiphop: [
    { pitch: 36, step: 0 },
    { pitch: 42, step: 2 },
    { pitch: 38, step: 4 },
    { pitch: 42, step: 6 },
    { pitch: 36, step: 10 },
    { pitch: 38, step: 12 },
    { pitch: 42, step: 14 },
  ],
}

/**
 * Create a seed sequence for drum generation.
 */
function createDrumSeed(
  style: string,
): mm.INoteSequence {
  const seedNotes = DRUM_SEED_NOTES[style as keyof typeof DRUM_SEED_NOTES] ??
    DRUM_SEED_NOTES.rock

  return {
    notes: seedNotes.map((n) => ({
      pitch: n.pitch,
      quantizedStartStep: n.step,
      quantizedEndStep: n.step + 1,
      velocity: 80 + Math.floor(Math.random() * 20),
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
  const sequence = await model.continueSequence(
    seed,
    stepsToGenerate,
    temperature,
  )
  const genTime = ((performance.now() - startTime) / 1000).toFixed(2)

  const notes = noteSequenceToNotes(sequence, startTick)
  logGenerate(`Generated ${notes.length} drum notes in ${genTime}s`)
  logGenerate("Sample notes:", notes.slice(0, 5).map(n => `pitch:${n.pitch}@${n.start}`))

  return notes
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

  const model = await getMusicVaeDrums()
  const numSamples = Math.ceil(bars / 4) // VAE generates 4 bars at a time

  logGenerate(`Sampling ${numSamples} x 4-bar patterns from latent space...`)

  const startTime = performance.now()
  const sequences = await model.sample(numSamples, temperature)
  const genTime = ((performance.now() - startTime) / 1000).toFixed(2)

  const allNotes: AppNote[] = []

  sequences.forEach((seq, i) => {
    const offset = startTick + i * 4 * TICKS_PER_QUARTER * 4 // 4 bars offset
    const notes = noteSequenceToNotes(seq, offset)
    logGenerate(`Sample ${i + 1}: ${notes.length} notes`)
    allNotes.push(...notes)
  })

  // Trim to requested bars
  const maxTick = startTick + bars * TICKS_PER_QUARTER * 4
  const trimmedNotes = allNotes.filter((n) => n.start < maxTick)

  logGenerate(`Generated ${trimmedNotes.length} drum notes total in ${genTime}s (VAE)`)

  return trimmedNotes
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
}

/**
 * Create a seed melody in a given scale.
 */
function createMelodySeed(
  scale: string,
  rangeLow: number,
): mm.INoteSequence {
  // Extract root from scale name (e.g., "C major" -> C, "Am" -> A)
  const rootMatch = scale.match(/^([A-G][#b]?)/)
  const rootName = rootMatch ? rootMatch[1] : "C"

  const noteMap: Record<string, number> = {
    C: 0, "C#": 1, Db: 1, D: 2, "D#": 3, Eb: 3, E: 4, F: 5,
    "F#": 6, Gb: 6, G: 7, "G#": 8, Ab: 8, A: 9, "A#": 10, Bb: 10, B: 11,
  }

  const rootPitch = noteMap[rootName] ?? 0
  const basePitch = Math.max(rangeLow, 60 + rootPitch - 12)

  // Determine scale intervals
  const isMinor = scale.toLowerCase().includes("minor") ||
    scale.toLowerCase().includes("m") && !scale.toLowerCase().includes("maj")
  const intervals = isMinor
    ? [0, 2, 3, 5, 7, 8, 10] // Natural minor
    : [0, 2, 4, 5, 7, 9, 11] // Major

  // Create simple ascending seed
  const seedNotes = [0, 2, 4, 2].map((degree, i) => ({
    pitch: basePitch + intervals[degree % intervals.length],
    quantizedStartStep: i * 4,
    quantizedEndStep: i * 4 + 4,
    velocity: 80,
  }))

  return {
    notes: seedNotes,
    totalQuantizedSteps: 16,
    quantizationInfo: { stepsPerQuarter: MAGENTA_STEPS_PER_QUARTER },
  }
}

/**
 * Generate melody using MelodyRNN.
 */
export async function generateMelodyMagenta(
  options: MelodyGenerationOptions,
): Promise<AppNote[]> {
  const { scale, bars, temperature, startTick, rangeLow, rangeHigh } = options

  logGenerate("=== MELODY (MelodyRNN) ===")
  logGenerate(`Scale: ${scale}, Bars: ${bars}, Temperature: ${temperature}`)
  logGenerate(`Range: ${rangeLow}-${rangeHigh}, StartTick: ${startTick}`)

  const model = await getMelodyRnn()
  const seed = createMelodySeed(scale, rangeLow)
  const stepsToGenerate = bars * 16

  logGenerate(`Seed created with ${seed.notes?.length ?? 0} notes, generating ${stepsToGenerate} steps...`)

  const startTime = performance.now()
  const sequence = await model.continueSequence(
    seed,
    stepsToGenerate,
    temperature,
  )
  const genTime = ((performance.now() - startTime) / 1000).toFixed(2)

  // Filter notes to requested range
  let notes = noteSequenceToNotes(sequence, startTick)
  const beforeFilter = notes.length
  notes = notes.filter((n) => n.pitch >= rangeLow && n.pitch <= rangeHigh)

  logGenerate(`Generated ${notes.length} melody notes in ${genTime}s (${beforeFilter - notes.length} filtered out of range)`)
  logGenerate("Sample notes:", notes.slice(0, 5).map(n => `pitch:${n.pitch}@${n.start}`))

  return notes
}

/**
 * Generate melody using MusicVAE (more varied).
 */
export async function generateMelodyVAE(
  options: MelodyGenerationOptions,
): Promise<AppNote[]> {
  const { bars, temperature, startTick, rangeLow, rangeHigh, scale } = options

  logGenerate("=== MELODY (MusicVAE) ===")
  logGenerate(`Scale hint: ${scale}, Bars: ${bars}, Temperature: ${temperature}`)
  logGenerate(`Range: ${rangeLow}-${rangeHigh}, StartTick: ${startTick}`)

  const model = await getMusicVaeMelody()
  const numSamples = Math.ceil(bars / 4)

  logGenerate(`Sampling ${numSamples} x 4-bar patterns from latent space...`)

  const startTime = performance.now()
  const sequences = await model.sample(numSamples, temperature)
  const genTime = ((performance.now() - startTime) / 1000).toFixed(2)

  const allNotes: AppNote[] = []

  sequences.forEach((seq, i) => {
    const offset = startTick + i * 4 * TICKS_PER_QUARTER * 4
    const notes = noteSequenceToNotes(seq, offset)
    logGenerate(`Sample ${i + 1}: ${notes.length} notes`)
    allNotes.push(...notes)
  })

  // Trim and filter
  const maxTick = startTick + bars * TICKS_PER_QUARTER * 4
  const trimmedNotes = allNotes
    .filter((n) => n.start < maxTick)
    .filter((n) => n.pitch >= rangeLow && n.pitch <= rangeHigh)

  logGenerate(`Generated ${trimmedNotes.length} melody notes total in ${genTime}s (VAE)`)

  return trimmedNotes
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

  // Create chord sequence in Magenta format
  const stepsPerChord = ticksToSteps(ticksPerChord)
  const chordSequence = chordProgression.flatMap((chord) => {
    // Repeat chord for each step it spans
    return Array(stepsPerChord).fill(chord)
  })

  logGenerate(`Chord sequence: ${stepsPerChord} steps per chord, ${chordSequence.length} total steps`)

  // Create minimal seed
  const seed: mm.INoteSequence = {
    notes: [
      { pitch: 60, quantizedStartStep: 0, quantizedEndStep: 4, velocity: 80 },
    ],
    totalQuantizedSteps: 4,
    quantizationInfo: { stepsPerQuarter: MAGENTA_STEPS_PER_QUARTER },
  }

  const stepsToGenerate = bars * 16

  logGenerate(`Generating ${stepsToGenerate} steps of melody over chords...`)

  const startTime = performance.now()
  const sequence = await model.continueSequence(
    seed,
    stepsToGenerate,
    temperature,
    chordSequence.slice(0, stepsToGenerate),
  )
  const genTime = ((performance.now() - startTime) / 1000).toFixed(2)

  const notes = noteSequenceToNotes(sequence, startTick)

  logGenerate(`Generated ${notes.length} improv notes in ${genTime}s`)
  logGenerate("Sample notes:", notes.slice(0, 5).map(n => `pitch:${n.pitch}@${n.start}`))

  return notes
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

