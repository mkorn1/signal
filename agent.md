# Hybrid Agent Architecture

## Overview

```
User Prompt → Frontend (AIChat) → Backend (LangGraph Agent) → Tool Calls → Frontend Execution → MobX Store
                    ↑                                                              ↓
                    └──────────────────── Results ────────────────────────────────┘
```

**Why hybrid?** LLM reasoning runs on the backend (secure API keys), while tools execute on the frontend (real-time UI updates against MobX store).

## Flow

1. **User sends prompt** → `AIChat.tsx` calls `runAgentLoop()`
2. **Song state serialized** → `serializeSongState()` creates compact JSON with tempo, tracks, note IDs
3. **Request sent to backend** → `POST /api/agent/step` with prompt + context
4. **LangGraph agent reasons** → Uses `create_react_agent()` with `interrupt_before=["tools"]`
5. **Tool calls returned** → Backend pauses before execution, returns tool calls to frontend
6. **Frontend executes** → `toolExecutor.ts` runs tools against the live `Song` object
7. **Results sent back** → Loop continues until `done: true`

---

## Available Tools

### High-Level Composition Tools (Preferred)

These tools use **Google Magenta AI** for realistic, human-sounding output:

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `createChordProgression` | Generate chord voicings with voice leading | chords, style (block/arpeggiated/broken/spread) |
| `generateDrumPattern` | **AI drum patterns** (Magenta DrumsRNN/MusicVAE) | style, bars, **temperature**, useVAE |
| `generateBassline` | Create bass lines following chords | chordProgression, style (root/walking/syncopated) |
| `generateMelody` | **AI melodies** (Magenta MelodyRNN/ImprovRNN/MusicVAE) | scale, bars, **temperature**, **chordProgression**, useVAE |
| `createArpeggio` | Generate arpeggio patterns | chord, pattern (up/down/updown), rate, octaves |
| `applyHumanization` | Add velocity/timing variation | velocityVariation, timingVariation, swing |

**New Magenta Parameters:**
- `temperature`: AI creativity (0.5=conservative, 1.0=balanced, 2.0=experimental)
- `useVAE`: Use MusicVAE for more varied output (vs RNN models)
- `chordProgression`: For melody - generates notes that follow chord changes

### Low-Level Tools (for fine-tuning)

- **Creation**: `createTrack`, `addNotes`, `setTempo`, `setTimeSignature`
- **Note Editing**: `deleteNotes`, `updateNotes`, `transposeNotes`, `duplicateNotes`, `quantizeNotes`
- **Track Ops**: `deleteTrack`, `renameTrack`, `setTrackInstrument`, `setTrackVolume`, `setTrackPan`
- **Controllers**: `setController`, `setPitchBend`

---

## Pattern Generation Architecture

### Current: Magenta.js AI Generation (Default)

The system uses [Google Magenta.js](https://github.com/magenta/magenta-js) for AI-powered music generation, with preset patterns as fallback.

```
High-level tool call (e.g., generateDrumPattern)
        ↓
toolExecutor.ts checks useMagenta flag (default: true)
        ↓
    ┌───────────────────────────────────────┐
    │           Magenta.js Models           │
    │  ┌─────────┐  ┌─────────┐  ┌───────┐ │
    │  │DrumsRNN │  │MelodyRNN│  │Improv │ │
    │  │         │  │         │  │  RNN  │ │
    │  └────┬────┘  └────┬────┘  └───┬───┘ │
    │       │            │           │     │
    │  ┌────┴────┐  ┌────┴────┐  ┌───┴───┐ │
    │  │MusicVAE │  │MusicVAE │  │ Chord │ │
    │  │ (drums) │  │(melody) │  │Following│
    │  └─────────┘  └─────────┘  └───────┘ │
    └───────────────────────────────────────┘
        ↓
    NoteSequence → App format conversion
        ↓
    Notes added to track
```

### Magenta Models Used

| Model | Purpose | Parameters |
|-------|---------|------------|
| **DrumsRNN** | Drum pattern continuation from style seed | temperature, style |
| **MusicVAE (drums)** | Sample varied drum patterns from latent space | temperature |
| **MelodyRNN** | Melody continuation from scale seed | temperature, scale |
| **MusicVAE (melody)** | Sample varied melodies | temperature |
| **ImprovRNN** | Generate melody over chord progression | temperature, chordProgression |

### Temperature Parameter

Controls AI creativity/randomness:

| Value | Effect |
|-------|--------|
| 0.5 | Conservative, predictable patterns |
| 1.0 | Balanced (default) |
| 1.5 | More creative, varied |
| 2.0 | Very experimental |

### Model Loading & Caching

Models are lazy-loaded on first use and cached in memory:

```typescript
// magentaService.ts
const modelCache: ModelCache = {}

async function getDrumsRnn(): Promise<mm.MusicRNN> {
  return getModel("drumsRnn", async () => {
    const model = new mm.MusicRNN(CHECKPOINTS.drumsRnn)
    await model.initialize()
    return model
  })
}
```

**Model sizes:** ~5-15 MB per model, loaded from Google's CDN.

### Fallback: Preset Patterns

If Magenta fails or `useMagenta: false`, falls back to `musicGeneration.ts`:

**Supported Preset Styles:**
- **Drums**: rock, pop, jazz, funk, hiphop, latin, ballad, metal, electronic
- **Bass**: root, fifth, walking, arpeggiated, syncopated, octave, pedal
- **Chords**: block, arpeggiated, broken, spread
- **Melody**: arch, ascending, descending, wave, flat contours

### Future Enhancements

| Approach | Status | Notes |
|----------|--------|-------|
| **Magenta.js** | ✅ Implemented | Default for drums/melody |
| **Preset patterns** | ✅ Implemented | Fallback |
| **MIDI pattern library** | 🔜 Planned | Curated patterns from Groove/Lakh datasets |
| **External AI APIs** | Considered | AIVA, MusicGen for higher quality |

---

## Composition Workflow

The agent follows this workflow for creating songs:

1. **Plan the structure** - sections, key, tempo, instrumentation
2. **Build the foundation** - drums with `generateDrumPattern`, bass with `generateBassline`
3. **Add harmony** - chords with `createChordProgression`, arpeggios with `createArpeggio`
4. **Add melody** - lead lines with `generateMelody`
5. **Humanize and polish** - `applyHumanization` for natural feel

---

## MIDI Reference

- Note numbers: Middle C = 60, semitone = +1
- Timing: 480 ticks = quarter note
- Durations: whole=1920, half=960, quarter=480, eighth=240
- Velocity: 1-127

## Context Format

The agent receives song state before each request:

```
Current song state:
- Tempo: 120 BPM
- Time signature: 4/4
- Tracks: 2

Track details:
  [0] Conductor track (tempo/time signature)
  [1] Acoustic Grand Piano - channel 0, 16 notes
    Notes: [id:5 C4@0], [id:6 E4@480], [id:7 G4@960]...
```

---

## Key Files

| File | Purpose |
|------|---------|
| `backend/app/services/hybrid_agent.py` | LangGraph agent + system prompt + tool definitions |
| `app/src/services/hybridAgent/agentLoop.ts` | Frontend ↔ backend loop |
| `app/src/services/hybridAgent/toolExecutor.ts` | Tool → MobX store mapping |
| `app/src/services/hybridAgent/magentaService.ts` | **Magenta.js model loading, caching, generation** |
| `app/src/services/hybridAgent/musicGeneration.ts` | Preset patterns (fallback) |
| `app/src/services/hybridAgent/songStateSerializer.ts` | Song → JSON context |

