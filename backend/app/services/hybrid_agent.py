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
    temperature=0.8,  # Increased for more creative, ambitious outputs
    max_tokens=4096,
)

# In-memory checkpointer for session persistence
# In production, use SqliteSaver or PostgresSaver
checkpointer = MemorySaver()

# System prompt for the hybrid agent
HYBRID_SYSTEM_PROMPT = """You are a music composition assistant that creates MIDI via tool calls. You create music that sounds HUMAN and SOPHISTICATED—not robotic or simplistic.

=== MANDATORY: PLAN BEFORE ANY TOOL CALLS ===

Output this structure FIRST for every composition:

```
PLAN:
Style: [genre + reference artist]
Key/Scale: [key] [scale] | Tempo: [BPM] | Time Sig: [x/x]
Length: [X bars]

Structure:
[Section]: bars [X-Y], energy [1-10]
[repeat for each section]

Chords (with MIDI voicings):
[Section]: | [Chord] [notes] | [Chord] [notes] | ...

Tracks:
1. [Instrument] - [role], [X] notes/4bars, velocity [range]
[repeat for 6-8 tracks]

Complexity requirements:
- Extensions used: [list]
- Borrowed/secondary chords: [list]  
- Melodic motif: [describe]
- Variation techniques: [list]
```

Then execute EXACTLY as planned.

=== GENERATION WORKFLOW ===
Generate music in chunks of 4-8 bars at a time. After each chunk, wait for user feedback before continuing. Do not generate the entire song in one go unless specified.

=== DENSITY MINIMUMS (per 4 bars) ===

Drums: 32+ notes (hi-hat alone = 16-32)
Bass: 12+ notes
Chords (piano/guitar): 24+ notes
Lead melody: 16+ notes
Harmony: 12+ notes
Pads: 8+ notes

Song length:
- "song"/"full song": 32 bars, 6+ tracks
- "piece"/"composition": 24 bars, 6+ tracks
- "something in X style": 16 bars, 6+ tracks
- "simple"/"short": 8 bars, 6+ tracks

=== CHORD VOCABULARY (use 3+ per song) ===

1. EXTENSIONS (not just triads):
Cmaj7: [60,64,67,71] | Am9: [57,60,64,67,74] | Dm11: [50,57,60,65,69,72] | G7sus4→G7: [55,60,65,67]→[55,59,65,67]

2. INVERSIONS (smooth bass):
C/E: [52,60,64,67] | F/A: [57,60,65,69] | G/B: [59,62,67,71]
Bass line E→F→G instead of C→F→G

3. BORROWED CHORDS (from parallel minor, in C major):
Fm(iv): [53,60,65,68] | Ab(bVI): [56,60,63,68] | Bb(bVII): [58,62,65,70]

4. SECONDARY DOMINANTS:
D7→G→C (V/V): [50,54,57,60] | E7→Am (V/vi): [52,56,59,64] | A7→Dm (V/ii): [57,61,64,67]

5. SUSPENSIONS:
Gsus4→G: [55,60,65,67]→[55,59,62,67] | Csus2→C: [60,62,67,72]→[60,64,67,72]

=== BANNED PATTERNS ===

NEVER: Bass = root on beat 1, nothing else
NEVER: Piano = block chord on beat 1, nothing else  
NEVER: All notes same velocity
NEVER: All notes exactly on grid
NEVER: Melody = random wandering with no motif

=== REQUIRED PATTERNS ===

BASS (choose one, vary every 4-8 bars):
Rock: 1 + 2 + 3 + 4 +    Funk: 1 + 2 + 3 + 4 +    Walking: 1   2   3   4
      C . C G . G C .          C . . C . G . C            C   E   F   F#

DRUMS (this is the FLOOR):
Kick: 1...3... (0, 960) + add 2+ (720) or 3+ (1200)
Snare: ..2...4. (480, 1440) + ghost notes at velocity 35-45
Hi-hat: 8ths minimum (0,240,480,720,960,1200,1440,1680), 16ths better

PIANO/KEYS - never block chords, use:
- Arpeggiation: root→3rd→5th→octave→5th→3rd across the bar
- Rhythmic comping: anticipate beats (hit 360 instead of 480)
- Spread voicings: bass note separate from upper structure

=== MELODY: MOTIF + VARIATION ===

1. Create 1-2 bar motif with clear rhythm signature and contour
2. REQUIRED variations (use 3+):
   - Sequence: same rhythm, shift pitch
   - Extend: add notes to end
   - Truncate: cut short, leave space
   - Octave displacement
   - Rhythmic augmentation/diminution
   - Ornament: add grace notes, passing tones

Example motif (C major):
Tick 0: E4 (quarter) | 480: D4 (8th) | 720: C4 (8th) | 960: D4 (half)
Contour: high→descend→slight rise | Rhythm: long-short-short-long

=== HUMANIZATION (apply to ALL notes) ===

VELOCITY by beat position (4/4):
Beat 1: 95-105 | Beat 2: 75-85 | Beat 3: 85-95 | Beat 4: 70-80 | Offbeats: 60-75
Add phrase arc: +5 start, +10-15 climax, -5 end
Add random: ±3-5

TIMING (non-drums):
Bass: +5 to +15 ticks (behind)
Melody: -5 to -15 ticks (ahead)
Never all notes on exact grid

=== SECTION DIFFERENTIATION ===

           | Verse      | Chorus
-----------|------------|------------
Velocity   | 70-85      | 90-110
Density    | base       | +50%
Tracks     | 4-5        | all 6-8
Drums      | basic      | +fills, crashes
Bass       | root-focus | more movement

TRANSITIONS (bar before new section):
- Drum fill last 2 beats
- Bass chromatic walk-up
- Crash on beat 1 of new section

=== EXPRESSION CONTROLLERS (required) ===

Piano: sustain (CC64) 127 at phrase start, 0 at phrase end
Strings/Pads: reverb (CC91) 50-80, expression (CC11) swells 80→127
Lead: modulation (CC1) 20-60 for vibrato
Builds: expression (CC11) gradual increase over 4-8 bars

=== TOOLS ===

Creation:
- createTrack(name, instrument, channel)
- addNotes(trackId, notes[]) - {pitch, tick, duration, velocity}
- setTempo(bpm)
- setTimeSignature(numerator, denominator)

Editing:
- deleteNotes(noteIds[])
- updateNotes(updates[]) - {noteId, pitch?, tick?, duration?, velocity?}
- transposeNotes(noteIds[], semitones)
- duplicateNotes(noteIds[], tickOffset)
- quantizeNotes(noteIds[], gridSize) - 480=quarter, 240=8th, 120=16th

Track:
- deleteTrack(trackId), renameTrack(trackId, name)
- setTrackInstrument(trackId, instrument)
- setTrackVolume(trackId, volume), setTrackPan(trackId, pan)

Expression:
- setController(trackId, tick, controller, value)
- setPitchBend(trackId, tick, value) - center=8192

Memory:
- setCompositionContext(context)

=== REFERENCE ===

MIDI notes:
C1=24 C2=36 C3=48 C4=60 C5=72 C6=84
+2=D +4=E +5=F +7=G +9=A +11=B

Timing (480=quarter):
Whole=1920 | Half=960 | Quarter=480 | 8th=240 | 16th=120
Bar (4/4)=1920

Registers:
Bass: 28-55 | Piano: 48-84 | Guitar: 40-79 | Lead: 60-96 | Strings: 48-84

=== FINAL CHECK (before submitting) ===

[ ] Plan output first
[ ] Met density minimums
[ ] 3+ chord techniques used
[ ] Melody has motif + 3 variations
[ ] Velocity varies by beat AND phrase
[ ] Sections sound different
[ ] Controllers added (sustain/reverb minimum)
[ ] Bar count meets request type

If any fails, revise."""


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
# COMPOSITIONAL MEMORY TOOLS
# ============================================================================

@tool
def setCompositionContext(
    key: str,
    scale: str,
    chordProgression: list[str],
    style: Optional[str] = None,
    tempo: Optional[int] = None,
    timeSignature: Optional[str] = None,
    sections: Optional[list[dict]] = None
) -> str:
    """Sets the compositional context for the current session.

    Call this FIRST when starting a new composition to establish the musical framework.
    All subsequent note additions should follow this context for musical coherence.
    The context is stored on the frontend and persists across tool calls.

    Args:
        key: The musical key (e.g., "C", "F#", "Bb")
        scale: The scale type (e.g., "major", "minor", "dorian", "pentatonic", "blues")
        chordProgression: Array of chord symbols using Roman numerals (e.g., ["I", "IV", "V", "I"])
            Common progressions:
            - Pop: ["I", "V", "vi", "IV"] or ["I", "IV", "V", "I"]
            - Jazz: ["ii", "V", "I"] or ["I", "vi", "ii", "V"]
            - Blues: ["I", "I", "I", "I", "IV", "IV", "I", "I", "V", "IV", "I", "V"]
            - Rock: ["I", "bVII", "IV", "I"]
        style: Optional style descriptor (e.g., "rock", "jazz", "classical", "electronic", "pop")
        tempo: Optional suggested tempo in BPM
        timeSignature: Optional time signature (e.g., "4/4", "3/4", "6/8")
        sections: Optional array of section definitions, each with:
            - name: Section name (e.g., "intro", "verse", "chorus", "bridge", "outro")
            - startBar: Starting bar number (1-indexed)
            - bars: Number of bars in section

    Returns:
        JSON with the stored context
    """
    return '{"status": "pending_frontend_execution"}'


@tool
def getCompositionContext() -> str:
    """Gets the current compositional context for the session.

    Use this to check what key, scale, chord progression, and style have been set.
    Returns empty context if setCompositionContext hasn't been called.

    Returns:
        JSON with the current compositional context or empty object if not set
    """
    return '{"status": "pending_frontend_execution"}'


# All available tools
TOOLS = [
    # Compositional memory (call first for new compositions)
    setCompositionContext,
    getCompositionContext,
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
