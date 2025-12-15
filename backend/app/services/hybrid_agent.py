"""Hybrid agent service - runs LLM reasoning on backend, tools execute on frontend.

Uses DeepAgents with interrupt_before to pause before tool execution,
returning tool calls to the frontend for execution against the MobX store.
"""

import uuid
from typing import Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from app.config import get_settings

settings = get_settings()

# Initialize LLM
model = ChatOpenAI(
    model=settings.openrouter_model,
    base_url="https://openrouter.ai/api/v1",
    api_key=settings.openrouter_api_key,
    default_headers={
        "HTTP-Referer": "https://github.com/signal-music-composer",
        "X-Title": "AI Music Composer",
    },
    temperature=0.7,
    max_tokens=4096,
)

# In-memory checkpointer for session persistence
# In production, use SqliteSaver or PostgresSaver
checkpointer = MemorySaver()

# System prompt for the hybrid agent
HYBRID_SYSTEM_PROMPT = """You are a professional music composition assistant that creates rich, realistic MIDI compositions by calling tools.

COMPOSITION PHILOSOPHY:
- Create music that sounds like it was made by a skilled human musician
- Use varied rhythms, dynamics, and articulations - avoid robotic, quantized patterns
- Layer multiple instruments for depth and texture
- Apply humanization to make parts feel alive
- Use the high-level composition tools whenever possible for better musical results

AVAILABLE TOOLS:

=== HIGH-LEVEL COMPOSITION TOOLS (PREFERRED) ===
These tools use Google Magenta AI for realistic, human-like musical output.

- createChordProgression: Generate chord voicings with voice leading
  Styles: "block", "arpeggiated", "broken", "spread"
  Example: createChordProgression(1, ["Cmaj7", "Am7", "Dm7", "G7"], style="arpeggiated")

- generateDrumPattern: AI-powered drum pattern generation (Magenta DrumsRNN/MusicVAE)
  Styles: "rock", "pop", "jazz", "funk", "hiphop", "latin", "ballad", "metal", "electronic"
  Temperature: 0.5 (conservative) to 2.0 (experimental), default 1.0
  useVAE: true for more varied patterns, false for style-specific (default)
  Example: generateDrumPattern(2, "jazz", bars=8, temperature=1.2)

- generateBassline: Create bass lines following chord progressions
  Styles: "root", "fifth", "walking", "arpeggiated", "syncopated", "octave", "pedal"
  Example: generateBassline(3, ["Cmaj7", "Am7"], style="walking")

- generateMelody: AI-powered melody generation (Magenta MelodyRNN/MusicVAE)
  Scales: "C major", "A minor", "D dorian", etc.
  Temperature: 0.5 (conservative) to 2.0 (experimental), default 1.0
  style: Genre hint for better results - "rock", "punk", "metal", "jazz", "pop", "ballad", "electronic", "funk", "hiphop", "latin", "country"
  chordProgression: Optional - melody will follow these chords
  IMPORTANT: Always pass the style parameter matching the song genre!
  Example: generateMelody(1, "A minor", bars=8, temperature=1.2, style="punk")

- createArpeggio: Generate arpeggio patterns from chords
  Patterns: "up", "down", "updown", "downup", "random"
  Example: createArpeggio(1, "Am7", pattern="updown", rate=120)

- applyHumanization: Make parts sound natural and human
  Example: applyHumanization(1, [], velocityVariation=15, swing=50)

=== LOW-LEVEL TOOLS (for fine-tuning) ===

Creation Tools:
- createTrack: Create a new track with an instrument
- addNotes: Add individual notes (use for specific passages, not full compositions)
- setTempo: Set the tempo in BPM
- setTimeSignature: Set the time signature

Note Editing Tools:
- deleteNotes: Remove notes by their IDs
- updateNotes: Modify note properties (pitch, timing, duration, velocity)
- transposeNotes: Shift notes up/down by semitones
- duplicateNotes: Copy notes with optional time offset
- quantizeNotes: Snap notes to a grid

Track Operation Tools:
- deleteTrack: Remove a track from the song
- renameTrack: Change a track's name
- setTrackInstrument: Change a track's instrument
- setTrackVolume: Set track volume (0-127)
- setTrackPan: Set stereo pan position (0=left, 64=center, 127=right)

Advanced Controller Tools:
- setController: Set any MIDI CC value (sustain pedal, modulation, reverb, etc.)
- setPitchBend: Set pitch bend (0-16383, center=8192)

IMPORTANT: When calling tools, you must use the exact parameter names and formats specified.

SONG STATE CONTEXT:
You will receive the current song state before each request. This tells you:
- Current tempo and time signature
- Existing tracks with their IDs, instruments, channels, and note counts
- Track [0] is usually the conductor track (tempo/time signature only)

The song state also includes note IDs that you can use for editing operations.

Use this context to:
- Reference existing tracks by their ID when adding or editing notes
- Reference note IDs when editing, deleting, transposing, or duplicating notes
- Avoid creating duplicate tracks (e.g., if a piano track exists, use it)
- Understand what's already in the song before making changes

Example context:
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

MIDI REFERENCE:
- Note numbers: Middle C = 60, each semitone = +1 (C4=60, D4=62, E4=64, F4=65, G4=67, A4=69, B4=71)
- Timing: 480 ticks = 1 quarter note
- Durations: whole=1920, half=960, quarter=480, eighth=240, sixteenth=120
- Velocity: 1-127 (loudness), typical range 60-100
- Common scales from C: Major [60,62,64,65,67,69,71,72], Minor [60,62,63,65,67,68,70,72]
- Quantize grid sizes: 480 (quarter), 240 (eighth), 120 (sixteenth), 60 (32nd)

CONTROLLER REFERENCE (for setController):
- "modulation" (CC1): Vibrato/tremolo depth, 0-127
- "volume" (CC7): Track volume, 0-127
- "pan" (CC10): Stereo position, 0=left, 64=center, 127=right
- "expression" (CC11): Dynamic expression, 0-127
- "sustain" (CC64): Sustain pedal, 0=off, 64+=on
- "reverb" (CC91): Reverb depth, 0-127
- "chorus" (CC93): Chorus depth, 0-127
- "brightness" (CC74): Filter cutoff, 0-127
- "attack" (CC73): Attack time, 0-127
- "release" (CC72): Release time, 0-127
Or use any CC number directly: "CC1", "CC64", "7", etc.

COMPOSITION WORKFLOW:
For creating new music, follow this approach:

1. PLAN THE STRUCTURE
   - Determine the song sections (intro, verse, chorus, bridge, outro)
   - Choose a key, tempo, and time signature
   - Decide on instrumentation

2. BUILD THE FOUNDATION
   - Set tempo and time signature
   - Create a drum track and use generateDrumPattern for the rhythm foundation
   - Create a bass track and use generateBassline following your chord progression

3. ADD HARMONY
   - Create a chord instrument track (piano, guitar, or pads)
   - Use createChordProgression for rich, properly-voiced chords
   - Consider using createArpeggio for texture and movement

4. ADD MELODY
   - Create a lead instrument track
   - Use generateMelody as a starting point
   - Refine with updateNotes if needed

5. HUMANIZE AND POLISH
   - Apply applyHumanization to drum, bass, and other parts
   - Add swing where appropriate (jazz, funk, R&B)
   - Adjust velocities for dynamics

EDITING TIPS:
- To change wrong notes: use updateNotes to fix pitch, or deleteNotes + addNotes
- To shift timing: use updateNotes with new tick values
- To make louder/softer: use updateNotes to change velocity
- To move to different octave: use transposeNotes with semitones=12 or -12
- To extend/repeat a phrase: use duplicateNotes
- To fix timing issues: use quantizeNotes with appropriate grid size

CONTROLLER TIPS:
- For piano sustain: setController with "sustain", value=127 (on) or 0 (off)
- For vibrato: setController with "modulation", value 0-127
- For pitch slides: use setPitchBend at different ticks (8192=center, 0=down, 16383=up)
- Controllers can change over time: call setController at different tick positions

SIMPLE REQUESTS:
For simple requests (e.g., "add a piano track", "transpose up an octave"), execute tools directly without extensive planning.

IMPORTANT - CONVERSATION MEMORY:
- This is a multi-turn conversation. ALWAYS remember what the user told you earlier.
- If the user mentioned a style, genre, key, tempo preference, or any other detail earlier in the conversation, REMEMBER IT and apply it to all subsequent actions.
- NEVER ask the user to repeat information they already provided. If they said "rock song" or "jazz style" earlier, that context applies to the whole session.
- Reference earlier conversation when relevant: "Based on the rock style you mentioned..."

Be concise in your responses. Focus on helping the user create great music."""


# Tool definitions that match the frontend schemas
# These are "dummy" tools - they just return a placeholder since actual execution happens on frontend

@tool
def createTrack(instrumentName: str, trackName: Optional[str] = None) -> str:
    """Creates a new MIDI track with the specified instrument.

    Args:
        instrumentName: The instrument to use. GM names like "Acoustic Grand Piano" or aliases like "piano", "guitar", "drums", "bass"
        trackName: Optional custom name for the track. Defaults to the instrument name.

    Returns:
        JSON with trackId, instrumentName, programNumber, channel, isDrums
    """
    # This will be intercepted - actual execution on frontend
    return '{"trackId": 1, "status": "pending_frontend_execution"}'


@tool
def addNotes(trackId: int, notes: list[dict]) -> str:
    """Adds notes to an existing track.

    Args:
        trackId: The track ID returned from createTrack
        notes: Array of notes, each with: pitch (0-127, middle C=60), start (ticks, 480=quarter), duration (ticks), velocity (1-127, optional, default 100)

    Returns:
        JSON with trackId and noteCount
    """
    return '{"status": "pending_frontend_execution"}'


@tool
def setTempo(bpm: int, tick: int = 0) -> str:
    """Sets the tempo (BPM) at a specific position in the song.

    Args:
        bpm: Beats per minute (20-300). Common: Andante 76-108, Moderato 108-120, Allegro 120-168
        tick: Position in ticks where tempo takes effect. Default: 0 (start)

    Returns:
        JSON with bpm and tick
    """
    return '{"status": "pending_frontend_execution"}'


@tool
def setTimeSignature(numerator: int, denominator: int, tick: int = 0) -> str:
    """Sets the time signature at a specific position.

    Args:
        numerator: Beats per measure (1-16). Common: 4 for 4/4, 3 for 3/4
        denominator: Beat unit: 2=half, 4=quarter, 8=eighth, 16=sixteenth
        tick: Position in ticks where time signature takes effect. Default: 0

    Returns:
        JSON with numerator, denominator, and tick
    """
    return '{"status": "pending_frontend_execution"}'


# ============================================================================
# NOTE EDITING TOOLS
# ============================================================================

@tool
def deleteNotes(trackId: int, noteIds: list[int]) -> str:
    """Deletes notes from a track by their IDs.

    Args:
        trackId: The track ID containing the notes
        noteIds: Array of note IDs to delete

    Returns:
        JSON with trackId and deletedCount
    """
    return '{"status": "pending_frontend_execution"}'


@tool
def updateNotes(trackId: int, updates: list[dict]) -> str:
    """Updates properties of existing notes.

    Args:
        trackId: The track ID containing the notes
        updates: Array of update objects, each with: id (required), and optional: pitch (0-127), tick (position), duration (ticks), velocity (1-127)

    Returns:
        JSON with trackId and updatedCount
    """
    return '{"status": "pending_frontend_execution"}'


@tool
def transposeNotes(trackId: int, noteIds: list[int], semitones: int) -> str:
    """Transposes notes by a number of semitones.

    Args:
        trackId: The track ID containing the notes
        noteIds: Array of note IDs to transpose
        semitones: Number of semitones to transpose (positive = up, negative = down). Range: -127 to 127

    Returns:
        JSON with trackId, transposedCount, and semitones
    """
    return '{"status": "pending_frontend_execution"}'


@tool
def duplicateNotes(trackId: int, noteIds: list[int], offsetTicks: int = 0) -> str:
    """Duplicates notes with an optional time offset.

    Args:
        trackId: The track ID containing the notes
        noteIds: Array of note IDs to duplicate
        offsetTicks: Tick offset for the duplicated notes. Default 0 places them immediately after the originals.

    Returns:
        JSON with trackId, duplicatedCount, newNoteIds, and actualOffset
    """
    return '{"status": "pending_frontend_execution"}'


@tool
def quantizeNotes(trackId: int, noteIds: list[int], gridSize: int) -> str:
    """Quantizes notes to snap to a grid.

    Args:
        trackId: The track ID containing the notes
        noteIds: Array of note IDs to quantize
        gridSize: Grid size in ticks. Common values: 480 (quarter), 240 (eighth), 120 (sixteenth), 60 (32nd)

    Returns:
        JSON with trackId and quantizedCount
    """
    return '{"status": "pending_frontend_execution"}'


# ============================================================================
# TRACK OPERATION TOOLS
# ============================================================================

@tool
def deleteTrack(trackId: int) -> str:
    """Deletes a track from the song.

    Args:
        trackId: The track ID to delete. Cannot delete the conductor track (track 0).

    Returns:
        JSON with deletedTrackId and success status
    """
    return '{"status": "pending_frontend_execution"}'


@tool
def renameTrack(trackId: int, name: str) -> str:
    """Renames a track.

    Args:
        trackId: The track ID to rename
        name: The new name for the track

    Returns:
        JSON with trackId and newName
    """
    return '{"status": "pending_frontend_execution"}'


@tool
def setTrackInstrument(trackId: int, instrumentName: str) -> str:
    """Changes the instrument of a track.

    Args:
        trackId: The track ID to modify
        instrumentName: The instrument to use. GM names like "Acoustic Grand Piano" or aliases like "piano", "guitar", "strings"

    Returns:
        JSON with trackId, instrumentName, and programNumber
    """
    return '{"status": "pending_frontend_execution"}'


@tool
def setTrackVolume(trackId: int, volume: int, tick: int = 0) -> str:
    """Sets the volume of a track.

    Args:
        trackId: The track ID to modify
        volume: Volume level 0-127 (0 = silent, 127 = max). Typical range: 80-100
        tick: Position in ticks where volume takes effect. Default: 0 (start)

    Returns:
        JSON with trackId, volume, and tick
    """
    return '{"status": "pending_frontend_execution"}'


@tool
def setTrackPan(trackId: int, pan: int, tick: int = 0) -> str:
    """Sets the stereo pan position of a track.

    Args:
        trackId: The track ID to modify
        pan: Pan position 0-127 (0 = full left, 64 = center, 127 = full right)
        tick: Position in ticks where pan takes effect. Default: 0 (start)

    Returns:
        JSON with trackId, pan, and tick
    """
    return '{"status": "pending_frontend_execution"}'


# ============================================================================
# ADVANCED CONTROLLER TOOLS
# ============================================================================

@tool
def setController(trackId: int, controllerType: str, value: int, tick: int = 0) -> str:
    """Sets any MIDI controller (CC) value on a track.

    This is a generic tool for all 128 MIDI CC controllers. Use friendly names
    or CC numbers directly.

    Args:
        trackId: The track ID to modify
        controllerType: Controller name or CC number. Common names:
            - "modulation" or "mod" (CC1) - vibrato/modulation depth
            - "breath" (CC2) - breath controller
            - "foot" (CC4) - foot controller
            - "volume" (CC7) - main volume
            - "pan" (CC10) - stereo position
            - "expression" (CC11) - dynamic expression
            - "sustain" or "hold" (CC64) - sustain pedal (0=off, 64+=on)
            - "soft" (CC67) - soft pedal
            - "reverb" (CC91) - reverb depth
            - "chorus" (CC93) - chorus depth
            - "brightness" (CC74) - filter cutoff
            - "attack" (CC73) - attack time
            - "release" (CC72) - release time
            Or use CC numbers: "CC1", "CC64", "7", etc.
        value: Controller value 0-127
        tick: Position in ticks where controller takes effect. Default: 0

    Returns:
        JSON with trackId, controllerType, controllerNumber, value, and tick
    """
    return '{"status": "pending_frontend_execution"}'


@tool
def setPitchBend(trackId: int, value: int, tick: int = 0) -> str:
    """Sets pitch bend on a track.

    Pitch bend allows smooth pitch changes between notes. The range is 14-bit
    for fine control.

    Args:
        trackId: The track ID to modify
        value: Pitch bend value 0-16383. Center (no bend) = 8192.
            - 0 = maximum downward bend (typically -2 semitones)
            - 8192 = center/no bend
            - 16383 = maximum upward bend (typically +2 semitones)
        tick: Position in ticks where pitch bend takes effect. Default: 0

    Returns:
        JSON with trackId, value, and tick
    """
    return '{"status": "pending_frontend_execution"}'


# ============================================================================
# HIGH-LEVEL COMPOSITION TOOLS
# ============================================================================

@tool
def createChordProgression(
    trackId: int,
    chords: list[str],
    startTick: int = 0,
    ticksPerChord: int = 1920,
    style: str = "block",
    octave: int = 4,
    velocity: int = 80
) -> str:
    """Creates a chord progression with proper voicings.

    This high-level tool generates musically coherent chord voicings with
    voice leading. Use this instead of manually adding individual notes
    when you want professional-sounding chord progressions.

    Args:
        trackId: The track ID to add chords to
        chords: Array of chord symbols. Supported formats:
            - Triads: "C", "Cm", "Cdim", "Caug"
            - Sevenths: "Cmaj7", "Cm7", "C7", "Cdim7", "Cm7b5"
            - Extended: "Cmaj9", "C9", "Cm9", "C11", "C13"
            - Altered: "C7#9", "C7b9", "C7#11"
            - Slash chords: "C/E", "Am/G"
            - Any root note: C, C#, Db, D, D#, Eb, E, F, F#, Gb, G, G#, Ab, A, A#, Bb, B
        startTick: Starting position in ticks. Default: 0
        ticksPerChord: Duration of each chord in ticks. Default: 1920 (1 bar in 4/4)
        style: Voicing style:
            - "block": All notes together (default)
            - "arpeggiated": Notes played sequentially up
            - "broken": Alternating bass and upper notes
            - "spread": Wide voicing across octaves
        octave: Base octave for the chord root (2-6). Default: 4
        velocity: Note velocity 1-127. Default: 80

    Returns:
        JSON with trackId, noteCount, chordCount, and notes array with IDs

    Example:
        createChordProgression(1, ["Cmaj7", "Am7", "Dm7", "G7"], style="arpeggiated")
        Creates a ii-V-I jazz progression with arpeggiated voicings
    """
    return '{"status": "pending_frontend_execution"}'


@tool
def generateDrumPattern(
    trackId: int,
    style: str,
    bars: int = 4,
    startTick: int = 0,
    variation: str = "medium",
    includeFills: bool = True,
    swing: int = 0,
    temperature: float = 1.0,
    useMagenta: bool = True,
    useVAE: bool = False
) -> str:
    """Generates a drum pattern using AI (Magenta) or preset patterns.

    This tool creates complete drum patterns using Google's Magenta AI models
    trained on real drum performances. Falls back to preset patterns if needed.

    Args:
        trackId: The drum track ID (must be a drum/rhythm track)
        style: Drum pattern style hint for AI generation:
            - "rock": Standard rock beat
            - "pop": Pop/dance beat
            - "jazz": Jazz swing pattern
            - "funk": Syncopated funk groove
            - "hiphop": Hip-hop/trap style
            - "latin": Latin percussion
            - "ballad": Slow, sparse pattern
            - "metal": Double-kick metal
            - "electronic": EDM style
        bars: Number of bars to generate (1-16). Default: 4
        startTick: Starting position in ticks. Default: 0
        variation: Variation level for preset fallback. Default: "medium"
        includeFills: Add fills (preset fallback only). Default: true
        swing: Swing amount 0-100 (preset fallback only). Default: 0
        temperature: AI creativity level 0.5-2.0. Default: 1.0
            - 0.5: Conservative, predictable patterns
            - 1.0: Balanced (default)
            - 1.5: More creative, varied
            - 2.0: Very experimental
        useMagenta: Use Magenta AI for generation. Default: true
        useVAE: Use MusicVAE model (more varied) vs DrumsRNN. Default: false

    Returns:
        JSON with trackId, noteCount, bars, style, generationMethod

    Example:
        generateDrumPattern(2, "jazz", bars=8, temperature=1.2)
        Creates 8 bars of AI-generated jazz drums with moderate creativity
    """
    return '{"status": "pending_frontend_execution"}'


@tool
def generateBassline(
    trackId: int,
    chordProgression: list[str],
    startTick: int = 0,
    ticksPerChord: int = 1920,
    style: str = "root",
    octave: int = 2,
    velocity: int = 90
) -> str:
    """Generates a bassline following a chord progression.

    This high-level tool creates bass lines that follow chord roots with
    style-appropriate patterns. Use this instead of manually programming
    bass notes for more musical results.

    Args:
        trackId: The bass track ID
        chordProgression: Array of chord symbols (same format as createChordProgression)
        startTick: Starting position in ticks. Default: 0
        ticksPerChord: Duration of each chord in ticks. Default: 1920 (1 bar)
        style: Bassline style:
            - "root": Simple root notes on downbeats
            - "fifth": Root and fifth pattern
            - "walking": Jazz walking bass (chromatic approaches)
            - "arpeggiated": Arpeggiated chord tones
            - "syncopated": Funk/R&B syncopated pattern
            - "octave": Root with octave jumps
            - "pedal": Sustained pedal tone
        octave: Bass octave (1-3). Default: 2
        velocity: Note velocity 1-127. Default: 90

    Returns:
        JSON with trackId, noteCount, and bassline details

    Example:
        generateBassline(3, ["Cmaj7", "Am7", "Dm7", "G7"], style="walking")
        Creates a walking bass line over the chord changes
    """
    return '{"status": "pending_frontend_execution"}'


@tool
def generateMelody(
    trackId: int,
    scale: str,
    bars: int = 4,
    startTick: int = 0,
    contour: str = "arch",
    density: str = "medium",
    range_low: int = 60,
    range_high: int = 84,
    velocity: int = 85,
    temperature: float = 1.0,
    useMagenta: bool = True,
    useVAE: bool = False,
    chordProgression: Optional[list[str]] = None,
    style: Optional[str] = None
) -> str:
    """Generates a melodic line using AI (Magenta) with style-aware patterns.

    This tool creates melodies using Google's Magenta AI models trained on
    real music. Uses style-specific seed patterns for better genre-appropriate
    output. Can generate melodies that follow chord progressions.

    Args:
        trackId: The track ID to add the melody to
        scale: Scale hint for AI generation:
            - Major scales: "C", "C major", "G major", etc.
            - Minor scales: "Am", "A minor", "E minor", etc.
            - Modes: "D dorian", "E phrygian", "F lydian", "G mixolydian"
            - Other: "C pentatonic", "A blues"
        bars: Number of bars to generate (1-16). Default: 4
        startTick: Starting position in ticks. Default: 0
        contour: Contour shape (preset fallback only). Default: "arch"
        density: Note density (preset fallback only). Default: "medium"
        range_low: Lowest MIDI note (default: 60 = C4)
        range_high: Highest MIDI note (default: 84 = C6)
        velocity: Base velocity 1-127. Default: 85
        temperature: AI creativity level 0.5-2.0. Default: 1.0
            - 0.5: Conservative, predictable melodies
            - 1.0: Balanced (default)
            - 1.5: More creative, varied
            - 2.0: Very experimental
        useMagenta: Use Magenta AI for generation. Default: true
        useVAE: Use MusicVAE (more varied) vs MelodyRNN. Default: false
        chordProgression: Optional chord progression for the melody to follow.
            When provided, uses ImprovRNN to generate melody over chords.
            Example: ["C", "Am", "F", "G"]
        style: Genre/style hint for AI seed generation. Options:
            - "rock", "punk", "metal", "jazz", "pop", "ballad"
            - "electronic", "funk", "hiphop", "latin", "country"
            Uses style-specific patterns to seed the AI for better results.

    Returns:
        JSON with trackId, noteCount, bars, scale, generationMethod

    Example:
        generateMelody(1, "A minor", bars=8, temperature=1.2, style="punk")
        Creates an 8-bar AI melody with punk-style characteristics
    """
    return '{"status": "pending_frontend_execution"}'


@tool
def applyHumanization(
    trackId: int,
    noteIds: list[int],
    velocityVariation: int = 10,
    timingVariation: int = 10,
    swing: int = 0
) -> str:
    """Applies humanization to make notes sound more natural and less robotic.

    This tool adds subtle variations to velocity and timing that mimic
    human performance. Use this on programmed parts to make them feel
    more alive and musical.

    Args:
        trackId: The track ID containing the notes
        noteIds: Array of note IDs to humanize (use empty array [] for all notes in track)
        velocityVariation: Random velocity variation amount 0-30. Default: 10
            - 0: No variation
            - 10: Subtle, natural variation (default)
            - 20: More dynamic, expressive
            - 30: Very dynamic, almost random
        timingVariation: Random timing shift in ticks 0-30. Default: 10
            - 0: Perfectly quantized
            - 10: Subtle timing variation (default)
            - 20: Loose, human feel
            - 30: Very loose timing
        swing: Swing amount 0-100 applied to off-beat notes. Default: 0
            - 0: Straight timing
            - 30: Light swing
            - 50: Medium swing (jazz feel)
            - 70: Heavy swing

    Returns:
        JSON with trackId, humanizedCount, and variation details

    Example:
        applyHumanization(1, [], velocityVariation=15, swing=50)
        Humanizes all notes on track 1 with medium swing feel
    """
    return '{"status": "pending_frontend_execution"}'


@tool
def createArpeggio(
    trackId: int,
    chord: str,
    startTick: int = 0,
    duration: int = 1920,
    pattern: str = "up",
    rate: int = 240,
    octaves: int = 1,
    velocity: int = 80
) -> str:
    """Creates an arpeggiated pattern from a chord.

    This tool generates arpeggio patterns that can be used for
    accompaniment, intros, or textural elements.

    Args:
        trackId: The track ID to add the arpeggio to
        chord: Chord symbol (same format as createChordProgression)
        startTick: Starting position in ticks. Default: 0
        duration: Total duration of the arpeggio in ticks. Default: 1920 (1 bar)
        pattern: Arpeggio pattern:
            - "up": Ascending (default)
            - "down": Descending
            - "updown": Up then down
            - "downup": Down then up
            - "random": Random order
            - "outside_in": Outer notes to inner
            - "inside_out": Inner notes to outer
        rate: Note rate in ticks (240=eighth, 120=sixteenth). Default: 240
        octaves: Number of octaves to span (1-3). Default: 1
        velocity: Note velocity 1-127. Default: 80

    Returns:
        JSON with trackId, noteCount, and arpeggio details

    Example:
        createArpeggio(1, "Am7", duration=3840, pattern="updown", rate=120)
        Creates a 2-bar sixteenth-note arpeggio on Am7
    """
    return '{"status": "pending_frontend_execution"}'


# All available tools
TOOLS = [
    # Creation tools
    createTrack,
    addNotes,
    setTempo,
    setTimeSignature,
    # Note editing tools
    deleteNotes,
    updateNotes,
    transposeNotes,
    duplicateNotes,
    quantizeNotes,
    # Track operation tools
    deleteTrack,
    renameTrack,
    setTrackInstrument,
    setTrackVolume,
    setTrackPan,
    # Advanced controller tools
    setController,
    setPitchBend,
    # High-level composition tools
    createChordProgression,
    generateDrumPattern,
    generateBassline,
    generateMelody,
    applyHumanization,
    createArpeggio,
]


def create_agent():
    """Create the hybrid agent with interrupt_before for tool execution."""
    agent = create_react_agent(
        model=model,
        tools=TOOLS,
        checkpointer=checkpointer,
        interrupt_before=["tools"],  # Pause before executing tools
        prompt=HYBRID_SYSTEM_PROMPT,
    )
    return agent


# Singleton agent instance
_agent = None


def get_agent():
    """Get or create the singleton agent instance."""
    global _agent
    if _agent is None:
        _agent = create_agent()
    return _agent


def generate_thread_id() -> str:
    """Generate a new thread ID for a session."""
    return str(uuid.uuid4())


async def start_agent_step(prompt: str, thread_id: Optional[str] = None, context: Optional[str] = None) -> dict:
    """Start a new agent interaction or continue an existing one.

    Args:
        prompt: The user's request
        thread_id: Optional existing thread ID to continue. If None, creates new session.
        context: Optional song state context to prepend to the prompt.

    Returns:
        dict with:
        - thread_id: Session identifier for continuation
        - tool_calls: List of tool calls to execute (if paused at interrupt)
        - done: True if agent completed without needing tool execution
        - message: Agent's response message (if done)
    """
    agent = get_agent()

    # Create or reuse thread ID
    if thread_id is None:
        thread_id = generate_thread_id()

    config = {"configurable": {"thread_id": thread_id}}

    # Load existing conversation history from checkpoint
    existing_state = await agent.aget_state(config)
    existing_messages = existing_state.values.get("messages", []) if existing_state.values else []

    # VERY VISIBLE LOGGING
    print(f"\n{'='*60}")
    print(f"[HYBRID_AGENT] thread_id: {thread_id}")
    print(f"[HYBRID_AGENT] existing_messages count: {len(existing_messages)}")
    print(f"[HYBRID_AGENT] existing_state.values keys: {existing_state.values.keys() if existing_state.values else 'None'}")
    print(f"{'='*60}\n")

    # Build the full message with context if provided
    full_prompt = prompt
    if context:
        full_prompt = f"{context}\n\n---\n\nUser request: {prompt}"

    # Build the message list - include history if continuing conversation
    new_message = {"role": "user", "content": full_prompt}
    if existing_messages:
        messages_to_send = {"messages": existing_messages + [new_message]}
        print(f"[HYBRID_AGENT] CONTINUING conversation with {len(existing_messages)} + 1 messages")
    else:
        messages_to_send = {"messages": [new_message]}
        print(f"[HYBRID_AGENT] STARTING new conversation")

    # Invoke the agent
    result = await agent.ainvoke(
        messages_to_send,
        config=config,
    )

    # Check if we're paused at interrupt (tool calls pending)
    state = await agent.aget_state(config)

    if state.next:  # There are pending nodes (we hit interrupt)
        # Extract tool calls from the last AI message
        last_message = result["messages"][-1]
        tool_calls = []

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            for tc in last_message.tool_calls:
                tool_calls.append({
                    "id": tc["id"],
                    "name": tc["name"],
                    "args": tc["args"],
                })

        return {
            "thread_id": thread_id,
            "tool_calls": tool_calls,
            "done": False,
            "message": None,
        }
    else:
        # Agent completed
        last_message = result["messages"][-1]
        content = last_message.content if hasattr(last_message, "content") else str(last_message)

        return {
            "thread_id": thread_id,
            "tool_calls": [],
            "done": True,
            "message": content,
        }


async def resume_agent_step(thread_id: str, tool_results: list[dict]) -> dict:
    """Resume agent after frontend tool execution.

    Args:
        thread_id: Session identifier from start_agent_step
        tool_results: List of tool results, each with:
            - id: Tool call ID from the original tool_calls
            - result: JSON string result from frontend execution

    Returns:
        Same format as start_agent_step
    """
    agent = get_agent()
    config = {"configurable": {"thread_id": thread_id}}

    # Build tool messages to resume with
    from langchain_core.messages import ToolMessage

    tool_messages = []
    for tr in tool_results:
        tool_messages.append(
            ToolMessage(
                content=tr["result"],
                tool_call_id=tr["id"],
            )
        )

    # Resume the agent with tool results
    result = await agent.ainvoke(
        Command(resume=tool_messages),
        config=config,
    )

    # Check state again
    state = await agent.aget_state(config)

    if state.next:  # More tool calls
        last_message = result["messages"][-1]
        tool_calls = []

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            for tc in last_message.tool_calls:
                tool_calls.append({
                    "id": tc["id"],
                    "name": tc["name"],
                    "args": tc["args"],
                })

        return {
            "thread_id": thread_id,
            "tool_calls": tool_calls,
            "done": False,
            "message": None,
        }
    else:
        last_message = result["messages"][-1]
        content = last_message.content if hasattr(last_message, "content") else str(last_message)

        return {
            "thread_id": thread_id,
            "tool_calls": [],
            "done": True,
            "message": content,
        }


async def stream_agent_step(prompt: str, thread_id: Optional[str] = None, context: Optional[str] = None):
    """Stream agent events as SSE.

    Yields events:
        - thinking: Agent reasoning/processing
        - tool_calls: Tools to execute on frontend
        - message: Final response from agent
        - error: Any errors that occurred

    Args:
        prompt: The user's request
        thread_id: Optional existing thread ID to continue
        context: Optional song state context

    Yields:
        dict with 'type' and event-specific data
    """
    agent = get_agent()

    # Create or reuse thread ID
    is_new_thread = thread_id is None
    if thread_id is None:
        thread_id = generate_thread_id()

    config = {"configurable": {"thread_id": thread_id}}

    # Load existing conversation history from checkpoint
    existing_state = await agent.aget_state(config)
    existing_messages = existing_state.values.get("messages", []) if existing_state.values else []
    print(f"[DEBUG] Thread {thread_id[:8]}... is_new={is_new_thread}, existing_messages={len(existing_messages)}")

    # Build the full message with context if provided
    full_prompt = prompt
    if context:
        full_prompt = f"{context}\n\n---\n\nUser request: {prompt}"

    # Build the message list - include history if continuing conversation
    new_message = {"role": "user", "content": full_prompt}
    if existing_messages:
        # Continue existing conversation - append new message to history
        # The existing_messages are LangChain message objects, we need to pass them through
        messages_to_send = {"messages": existing_messages + [new_message]}
        print(f"[DEBUG] Continuing conversation with {len(existing_messages)} existing + 1 new message")
    else:
        # New conversation
        messages_to_send = {"messages": [new_message]}
        print(f"[DEBUG] Starting new conversation")

    try:
        # Emit thinking event
        yield {"type": "thinking", "thread_id": thread_id, "content": "Processing your request..."}

        # Track which LLM run we've seen to avoid duplicate tokens
        seen_tokens = set()

        # Stream events from the agent
        async for event in agent.astream_events(
            messages_to_send,
            config=config,
            version="v2",
        ):
            event_type = event.get("event")
            run_id = event.get("run_id", "")

            # Handle LLM streaming tokens - only from chat model events
            if event_type == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    # Create a unique key for this token to avoid duplicates
                    token_key = f"{run_id}:{chunk.content}"
                    if token_key not in seen_tokens:
                        seen_tokens.add(token_key)
                        yield {"type": "thinking", "thread_id": thread_id, "content": chunk.content}

        # After streaming completes, check state for tool calls or completion
        state = await agent.aget_state(config)
        final_messages = state.values.get("messages", []) if state.values else []
        print(f"[DEBUG stream_agent_step] After stream: thread {thread_id[:8]}... has {len(final_messages)} messages")
        for i, msg in enumerate(final_messages):
            role = getattr(msg, 'type', 'unknown')
            content_preview = str(getattr(msg, 'content', ''))[:80]
            print(f"[DEBUG stream_agent_step]   [{i}] {role}: {content_preview}...")

        if state.next:
            # Agent is paused at interrupt - extract tool calls
            messages = state.values.get("messages", [])
            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    tool_calls = []
                    for tc in last_msg.tool_calls:
                        tool_calls.append({
                            "id": tc["id"],
                            "name": tc["name"],
                            "args": tc["args"],
                        })
                    yield {
                        "type": "tool_calls",
                        "thread_id": thread_id,
                        "tool_calls": tool_calls,
                        "done": False,
                    }
                    return

        # Agent completed - get final message
        messages = state.values.get("messages", [])
        if messages:
            last_message = messages[-1]
            content = last_message.content if hasattr(last_message, "content") else str(last_message)
            yield {
                "type": "message",
                "thread_id": thread_id,
                "content": content,
                "done": True,
            }

    except Exception as e:
        yield {"type": "error", "thread_id": thread_id, "error": str(e)}


async def stream_agent_resume(thread_id: str, tool_results: list[dict]):
    """Stream agent events after tool results are received.

    Yields events:
        - tool_results_received: Acknowledgment of tool results
        - thinking: Agent reasoning/processing
        - tool_calls: More tools to execute
        - message: Final response from agent
        - error: Any errors that occurred

    Args:
        thread_id: Session identifier from previous step
        tool_results: List of tool results from frontend

    Yields:
        dict with 'type' and event-specific data
    """
    from langchain_core.messages import ToolMessage

    agent = get_agent()
    config = {"configurable": {"thread_id": thread_id}}

    # Acknowledge tool results received
    yield {
        "type": "tool_results_received",
        "thread_id": thread_id,
        "count": len(tool_results),
    }

    # Build tool messages
    tool_messages = []
    for tr in tool_results:
        tool_messages.append(
            ToolMessage(
                content=tr["result"],
                tool_call_id=tr["id"],
            )
        )

    try:
        yield {"type": "thinking", "thread_id": thread_id, "content": "Processing tool results..."}

        # Track which LLM run we've seen to avoid duplicate tokens
        seen_tokens = set()

        # Stream events from the agent
        async for event in agent.astream_events(
            Command(resume=tool_messages),
            config=config,
            version="v2",
        ):
            event_type = event.get("event")
            run_id = event.get("run_id", "")

            # Handle LLM streaming tokens - only from chat model events
            if event_type == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    # Create a unique key for this token to avoid duplicates
                    token_key = f"{run_id}:{chunk.content}"
                    if token_key not in seen_tokens:
                        seen_tokens.add(token_key)
                        yield {"type": "thinking", "thread_id": thread_id, "content": chunk.content}

        # After streaming completes, check state for tool calls or completion
        state = await agent.aget_state(config)

        if state.next:
            # Agent is paused at interrupt - extract tool calls
            messages = state.values.get("messages", [])
            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    tool_calls = []
                    for tc in last_msg.tool_calls:
                        tool_calls.append({
                            "id": tc["id"],
                            "name": tc["name"],
                            "args": tc["args"],
                        })
                    yield {
                        "type": "tool_calls",
                        "thread_id": thread_id,
                        "tool_calls": tool_calls,
                        "done": False,
                    }
                    return

        # Agent completed - get final message
        messages = state.values.get("messages", [])
        if messages:
            last_message = messages[-1]
            content = last_message.content if hasattr(last_message, "content") else str(last_message)
            yield {
                "type": "message",
                "thread_id": thread_id,
                "content": content,
                "done": True,
            }

    except Exception as e:
        yield {"type": "error", "thread_id": thread_id, "error": str(e)}
