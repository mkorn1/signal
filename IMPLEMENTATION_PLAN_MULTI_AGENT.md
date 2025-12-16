# Multi-Agent Composition System - Implementation Plan

**Date:** 2025-12-15
**Project:** Signal Music Composer - Deep Agent Expansion
**Strategy:** Gradual Evolution (Hybrid Approach)

---

## Overview

This plan implements a hierarchical multi-agent composition system through **gradual, testable improvements** to the existing hybrid agent. Each phase delivers measurable quality gains while preserving the current chat-based UX.

**Core Principle:** Backend agents reason in musical abstractions (MIR); frontend executes MIDI operations (unchanged).

---

## Current State Analysis

### Existing Architecture

```
User Chat → Backend (hybrid_agent.py) → LLM decides tools → interrupt_before
                                              ↓
Frontend (toolExecutor.ts) executes 17 MIDI tools → Song (MobX) → UI updates
                                              ↓
                        Backend resumes with results → continues
```

**Strengths:**
- Working conversational interface with memory
- 17 MIDI tools for creation and editing
- Real-time UI updates via MobX
- Clean frontend-backend separation

**Limitations:**
- Single agent handles all musical decisions (arrangement + harmony + melody + rhythm)
- Outputs raw MIDI (tick math, note numbers) instead of musical concepts
- No iterative refinement or quality validation
- Inconsistent musical quality across sections

### Technology Stack

- **Backend:** Python, FastAPI, LangGraph, LangChain, OpenRouter (Claude)
- **Frontend:** TypeScript, React, MobX, Vite
- **Data:** Song class with Track arrays, MIDI events
- **Existing Endpoints:** `/api/agent/step`, `/api/agent/step/stream`

---

## What We're NOT Doing

To maintain focus and deliverability:

- ❌ Changing the chat interface UX
- ❌ Modifying the 17 existing MIDI tools
- ❌ Replacing the hybrid agent entirely
- ❌ Creating a separate "deep compose" UI mode
- ❌ Building a full DAW or audio engine
- ❌ Real-time audio synthesis changes

---

## Desired End State

### User Experience (Unchanged)

```
User: "Create a jazz ballad in Dm, 72 BPM"

[Chat interface, same as today]

Agent: "I'll create a jazz ballad for you..."
  → Style Agent determines: swing=0.55, extensions allowed, soft dynamics
  → Arranger creates: intro(8) → verse(16) → chorus(16) → outro(8)
  → Harmony generates: ii-V-I progressions with voice leading
  → Melody writes: singable phrases with jazz ornamentation
  → Rhythm creates: brushes pattern with swing feel
  → Orchestrator assigns: piano(chords), bass(walking), drums(brushes), sax(melody)
  → Critic validates: voice leading ✓, range ✓, style consistency ✓
  → MIR Compiler translates to existing MIDI tools
  → Frontend executes tools (same as today)

[Song appears with 4 tracks, musically coherent]
```

---

## Phase 1: MIR Foundation & Harmony Agent

### Overview

Build the Musical Intermediate Representation (MIR) schema and prove the concept by replacing direct MIDI generation with a **Harmony Agent** that thinks in chords, then compiles to MIDI.

### Success Metric

**Before:** Agent generates notes with tick math, occasional voice leading errors
**After:** Agent generates chord progressions with proper voice leading, compiled to MIDI

### Changes Required

#### 1.1 MIR Schema (Python Dataclasses)

**File:** `backend/app/services/mir/schema.py` (new)

```python
from dataclasses import dataclass
from typing import List, Literal, Optional

@dataclass
class Note:
    """Single note in MIR - uses music notation, not MIDI numbers."""
    pitch: str  # "D4", "F#3", "Bb2"
    bar: int
    beat: float  # 1.0 = downbeat, 1.5 = eighth note offset
    duration: str  # "whole", "half", "quarter", "eighth", "sixteenth"
    velocity: int = 80

@dataclass
class Chord:
    """Chord in MIR - harmonic thinking, not individual notes yet."""
    root: str  # "D", "F#", "Bb"
    quality: str  # "m9", "maj7", "7b9", "sus4", etc.
    bar: int
    beat: float
    duration: str
    voicing: List[str]  # ["D2", "A2", "F3", "C4", "E4"] - explicit pitches
    function: Optional[str] = None  # "tonic", "dominant", "subdominant", "passing"

@dataclass
class ChordProgression:
    """Container for harmony in a section."""
    track: str  # "piano", "guitar"
    section: str  # "verse_A", "chorus_B"
    chords: List[Chord]

@dataclass
class Section:
    """Song structure element."""
    name: str  # "intro", "verse_A", "chorus_B"
    bars: tuple[int, int]  # (start_bar, end_bar)
    key: str  # "Dm", "F", "A"
    tempo: int
    energy: str  # "soft", "building", "climax", "resolve"

@dataclass
class StyleGuide:
    """Style contract - all agents must follow this."""
    genre: str  # "jazz"
    subgenre: str  # "ballad"
    harmonic_complexity: str  # "complex" (9ths, 11ths), "medium" (7ths), "simple" (triads)
    swing: float  # 0.0 (straight), 0.55 (ballad), 0.67 (bebop)
    extensions_allowed: List[str]  # ["9", "11", "13", "b9", "#11"]
    tempo_range: tuple[int, int]
    reference_artists: List[str] = None
```

**Testing:**
- Unit tests: Serialize/deserialize MIR objects
- Validate chord quality parsing ("m9" → minor 9th)
- Validate pitch string parsing ("F#3" → MIDI 54)

#### 1.2 MIR → MIDI Compiler

**File:** `backend/app/services/mir/compiler.py` (new)

```python
from app.services.mir.schema import Chord, ChordProgression, Note
from typing import List, Dict

# Music theory constants
PITCH_TO_MIDI = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11
}

DURATION_TO_TICKS = {
    "whole": 1920, "half": 960, "quarter": 480,
    "eighth": 240, "sixteenth": 120, "thirtysecond": 60
}

def pitch_string_to_midi(pitch: str) -> int:
    """Convert 'D4' → 62, 'F#3' → 54."""
    note = pitch[:-1]  # "D", "F#"
    octave = int(pitch[-1])  # 4, 3
    return PITCH_TO_MIDI[note] + (octave + 1) * 12

def beats_to_ticks(bar: int, beat: float, timebase: int = 480) -> int:
    """Convert musical time (bar 2, beat 1.5) → tick position."""
    # Assumes 4/4 time - bar 1 = tick 0
    ticks_per_bar = timebase * 4
    return (bar - 1) * ticks_per_bar + int((beat - 1) * timebase)

def duration_to_ticks(duration: str) -> int:
    """Convert 'quarter' → 480 ticks."""
    return DURATION_TO_TICKS.get(duration, 480)

def compile_chord_to_notes(chord: Chord) -> List[Dict]:
    """
    Compile a Chord MIR object to MIDI tool call format.

    Returns list of notes for addNotes tool:
    [{"pitch": 62, "start": 0, "duration": 1920, "velocity": 75}, ...]
    """
    tick = beats_to_ticks(chord.bar, chord.beat)
    duration_ticks = duration_to_ticks(chord.duration)

    notes = []
    for pitch_str in chord.voicing:
        midi_pitch = pitch_string_to_midi(pitch_str)
        notes.append({
            "pitch": midi_pitch,
            "start": tick,
            "duration": duration_ticks,
            "velocity": chord.velocity if hasattr(chord, 'velocity') else 75
        })

    return notes

def compile_progression_to_tool_calls(
    progression: ChordProgression,
    track_id: int
) -> List[Dict]:
    """
    Compile ChordProgression → addNotes tool calls.

    Returns: [{"name": "addNotes", "args": {"trackId": 1, "notes": [...]}}]
    """
    all_notes = []
    for chord in progression.chords:
        all_notes.extend(compile_chord_to_notes(chord))

    # Sort by tick position
    all_notes.sort(key=lambda n: n["start"])

    return [{
        "name": "addNotes",
        "args": {
            "trackId": track_id,
            "notes": all_notes
        }
    }]
```

**Testing:**
- Unit test: `Chord("D", "m9", bar=1, beat=1, duration="whole", voicing=["D2","A2","F3","C4"])` → correct MIDI notes
- Integration test: Full chord progression → tool calls → execute on Song object

#### 1.3 Harmony Agent (LangGraph Subagent)

**File:** `backend/app/services/agents/harmony_agent.py` (new)

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from app.services.mir.schema import StyleGuide, Section, ChordProgression, Chord
from app.config import get_settings
import json

settings = get_settings()

HARMONY_SYSTEM_PROMPT = """You are the Harmony Agent, a specialist in chord progressions and voice leading.

ROLE:
- Generate chord progressions that fit the style guide and song structure
- Think in harmonic functions (tonic, dominant, subdominant, passing chords)
- Apply voice leading principles (smooth motion, avoid parallel fifths)
- Match harmonic rhythm to tempo and energy level

INPUT:
- Style Guide: genre, harmonic complexity, allowed extensions, swing feel
- Section: bars, key center, energy level

OUTPUT:
- ChordProgression object with explicit voicings in JSON format

HARMONIC PRINCIPLES:
1. Strong beats get stable chords (I, IV, V)
2. Weak beats allow passing chords
3. Voice leading: minimize movement, stepwise preferred
4. Match complexity to style (jazz allows 9ths/11ths/13ths, pop stays simpler)
5. Harmonic rhythm matches tempo (fast = slower changes, slow = more changes)
6. Cadences match section function (half cadence for question, authentic for answer)

VOICING RULES:
- Always provide explicit voicings as pitch strings ["D2", "A2", "F3", "C4"]
- Piano: 2-3 octave spread, bass notes in left hand (octave 2-3), chord tones in right (octave 3-5)
- Guitar: 1-2 octave spread, avoid wide stretches
- Check instrument range constraints

OUTPUT FORMAT (JSON):
{
  "track": "piano",
  "section": "verse_A",
  "chords": [
    {
      "root": "D",
      "quality": "m9",
      "bar": 1,
      "beat": 1.0,
      "duration": "whole",
      "voicing": ["D2", "A2", "F3", "C4", "E4"],
      "function": "tonic"
    },
    ...
  ]
}

Return ONLY valid JSON. No explanations."""

@tool
def generate_harmony_for_section(
    style_guide_json: str,
    section_json: str,
    track_name: str
) -> str:
    """Generate chord progression for a section. Returns ChordProgression JSON."""
    # This tool is a placeholder - actual generation happens in the agent
    return '{"status": "harmony_generated"}'

def create_harmony_agent():
    """Create the Harmony Agent."""
    model = ChatOpenAI(
        model=settings.openrouter_model,
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
        temperature=0.7,
    )

    agent = create_react_agent(
        model=model,
        tools=[generate_harmony_for_section],
        prompt=HARMONY_SYSTEM_PROMPT,
    )

    return agent

async def invoke_harmony_agent(
    style_guide: StyleGuide,
    section: Section,
    track_name: str = "piano"
) -> ChordProgression:
    """
    Invoke the harmony agent to generate a chord progression.

    Returns ChordProgression MIR object.
    """
    agent = create_harmony_agent()

    # Build the user prompt
    prompt = f"""Generate a chord progression for this section:

Style Guide:
{json.dumps(style_guide.__dict__, indent=2)}

Section:
{json.dumps(section.__dict__, indent=2)}

Track: {track_name}

Requirements:
- Use {style_guide.harmonic_complexity} harmony (extensions: {style_guide.extensions_allowed})
- Key: {section.key}
- Energy: {section.energy}
- Bars {section.bars[0]}-{section.bars[1]}
- Provide explicit voicings for {track_name}

Return ChordProgression JSON only."""

    result = await agent.ainvoke({"messages": [{"role": "user", "content": prompt}]})

    # Extract JSON from response
    last_message = result["messages"][-1].content

    # Parse the JSON response
    # In production, add error handling and validation
    progression_data = json.loads(last_message)

    # Convert to ChordProgression object
    chords = [Chord(**c) for c in progression_data["chords"]]
    progression = ChordProgression(
        track=progression_data["track"],
        section=progression_data["section"],
        chords=chords
    )

    return progression
```

**Testing:**
- Unit test: StyleGuide + Section → valid ChordProgression JSON
- Quality test: Check voice leading (max semitone movement between chords)
- Integration test: ChordProgression → compile → MIDI notes verify correct

#### 1.4 Integration into Hybrid Agent

**File:** `backend/app/services/hybrid_agent.py` (modify)

Add a new "smart tool" that uses the Harmony Agent internally:

```python
# Add to existing hybrid_agent.py

from app.services.agents.harmony_agent import invoke_harmony_agent
from app.services.mir.compiler import compile_progression_to_tool_calls
from app.services.mir.schema import StyleGuide, Section

@tool
def addChordProgression(
    trackId: int,
    key: str,
    bars: int,
    style: str = "jazz",
    harmonic_complexity: str = "medium",
    energy: str = "medium"
) -> str:
    """Add a chord progression to a track using intelligent harmony generation.

    This tool uses music theory to generate proper voice leading and harmonic rhythm.

    Args:
        trackId: The track ID (must be a harmonic instrument like piano, guitar, keys)
        key: Key signature (e.g., "Dm", "C", "F#m")
        bars: Number of bars to generate (4, 8, 16, 32)
        style: Musical style (jazz, pop, rock, classical)
        harmonic_complexity: "simple" (triads), "medium" (7ths), "complex" (9ths/11ths/13ths)
        energy: "soft", "medium", "high" - affects harmonic rhythm density

    Returns:
        JSON with trackId, bars generated, and chord count
    """
    # This will be intercepted and routed through harmony agent
    return '{"status": "pending_frontend_execution"}'

# Add to TOOLS list
TOOLS = [
    # ... existing tools ...
    addChordProgression,
]
```

**Modify the interrupt logic** to detect `addChordProgression` and route through subagent:

```python
# In hybrid_agent.py, after agent.ainvoke but before returning tool_calls

async def process_tool_calls(tool_calls: List[dict]) -> List[dict]:
    """Process tool calls, routing smart tools through subagents."""
    processed_calls = []

    for tc in tool_calls:
        if tc["name"] == "addChordProgression":
            # Route through Harmony Agent
            args = tc["args"]

            # Create minimal StyleGuide and Section
            style_guide = StyleGuide(
                genre=args.get("style", "jazz"),
                subgenre="",
                harmonic_complexity=args.get("harmonic_complexity", "medium"),
                swing=0.55,
                extensions_allowed=["7", "9", "11", "13"],
                tempo_range=(60, 140)
            )

            section = Section(
                name="generated",
                bars=(1, args["bars"]),
                key=args["key"],
                tempo=120,
                energy=args.get("energy", "medium")
            )

            # Invoke harmony agent
            progression = await invoke_harmony_agent(style_guide, section, "piano")

            # Compile to MIDI tool calls
            midi_calls = compile_progression_to_tool_calls(progression, args["trackId"])

            # Replace the smart tool with compiled MIDI tools
            processed_calls.extend([
                {"id": tc["id"] + f"_{i}", "name": call["name"], "args": call["args"]}
                for i, call in enumerate(midi_calls)
            ])
        else:
            # Pass through normal tools unchanged
            processed_calls.append(tc)

    return processed_calls

# In start_agent_step, before returning tool_calls:
if state.next:
    last_message = result["messages"][-1]
    tool_calls = []

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tc in last_message.tool_calls:
            tool_calls.append({
                "id": tc["id"],
                "name": tc["name"],
                "args": tc["args"],
            })

        # Process smart tools through subagents
        tool_calls = await process_tool_calls(tool_calls)

    return {
        "thread_id": thread_id,
        "tool_calls": tool_calls,
        "done": False,
        "message": None,
    }
```

### Success Criteria

#### Automated Verification:
- [x] MIR schema unit tests pass: `pytest backend/app/services/mir/test_schema.py`
- [x] Compiler tests pass: `pytest backend/app/services/mir/test_compiler.py`
- [x] Harmony agent generates valid JSON: `pytest backend/app/services/agents/test_harmony_agent.py`
- [x] Voice leading validation: no parallel fifths, max 5 semitone jumps
- [x] Backend starts without errors: `cd backend && uvicorn app.main:app`

#### Manual Verification:
- [ ] User: "Create a piano track with a jazz chord progression in Dm, 8 bars"
  - Agent calls `addChordProgression` internally
  - Piano track appears with ii-V-I progression
  - Chords have proper voice leading (verify in piano roll - notes move smoothly)
- [ ] Compare to old behavior: before Phase 1, same prompt generates simpler triads with jumpier voice leading

**Implementation Note:** After completing this phase and all automated tests pass, manually test the chord progression quality before proceeding to Phase 2.

---

## Phase 2: Style Agent & Arranger Agent

### Overview

Add **Style Agent** to establish musical DNA before generation, and **Arranger Agent** to create song structure. These become the first two steps in any composition request.

### Success Metric

**Before:** User says "create a jazz ballad" → agent generates music with inconsistent style
**After:** Style Agent creates binding contract → Arranger creates structure → consistent style throughout

### Changes Required

#### 2.1 Style Agent

**File:** `backend/app/services/agents/style_agent.py` (new)

```python
from app.services.mir.schema import StyleGuide
from langchain_openai import ChatOpenAI
import json

STYLE_SYSTEM_PROMPT = """You are the Style Agent. Your job is to establish the stylistic DNA for a composition.

ROLE:
- Parse user requests to identify genre, subgenre, and stylistic elements
- Create a binding StyleGuide that all downstream agents must follow
- Prevent style drift across sections

INPUT: User's style description (e.g., "jazz ballad", "upbeat funk", "melancholic indie rock")

OUTPUT: StyleGuide JSON with these decisions:
- genre/subgenre classification
- harmonic language (what extensions/chromaticism are allowed)
- rhythmic feel (straight, swing amount, syncopation level)
- melodic character (range, intervallic preference)
- dynamic range and energy curve
- reference artists (optional)

GENRE KNOWLEDGE:
- Jazz: harmonic complexity high, extensions (9,11,13), swing 0.55-0.67, sophisticated voice leading
- Pop: harmonic complexity medium, extensions (sus4, add9), straight feel, memorable hooks
- Rock: power chords, pentatonic scales, straight 8ths/16ths, guitar-driven
- Classical: functional harmony, clear forms, dynamic contrast
- Electronic: synth timbres, repetitive patterns, build/drop structure

OUTPUT FORMAT (JSON only, no explanation):
{
  "genre": "jazz",
  "subgenre": "ballad",
  "harmonic_complexity": "complex",
  "swing": 0.55,
  "extensions_allowed": ["7", "9", "11", "13", "b9", "#11"],
  "syncopation_level": "medium",
  "melodic_range": "C4-G5",
  "dynamic_range": "soft",
  "tempo_range": [60, 85],
  "reference_artists": ["Bill Evans", "Chet Baker"]
}
"""

async def invoke_style_agent(user_style_description: str) -> StyleGuide:
    """Invoke Style Agent to create StyleGuide from user description."""
    settings = get_settings()
    model = ChatOpenAI(
        model=settings.openrouter_model,
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
        temperature=0.7,
    )

    messages = [
        {"role": "system", "content": STYLE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Create a StyleGuide for: {user_style_description}"}
    ]

    response = await model.ainvoke(messages)
    style_data = json.loads(response.content)

    return StyleGuide(**style_data)
```

#### 2.2 Arranger Agent

**File:** `backend/app/services/agents/arranger_agent.py` (new)

```python
from app.services.mir.schema import StyleGuide, Section
from typing import List
import json

ARRANGER_SYSTEM_PROMPT = """You are the Arranger Agent. You define song structure - NO NOTES YET.

ROLE:
- Create the structural skeleton: form, sections, key centers, energy arc
- All other agents work within your scaffold

INPUT:
- StyleGuide (genre, tempo range, etc.)
- User length preference (optional)

OUTPUT: List of Section objects defining the song form

FORM TEMPLATES:
- Pop/Rock: intro(4-8) → verse(8-16) → chorus(8-16) → verse → chorus → bridge(8) → chorus → outro(4-8)
- Jazz: intro(8) → head(32 AABA) → solo(32) → head → outro(8)
- Electronic: intro(16) → buildup(16) → drop(32) → breakdown(16) → drop(32) → outro(16)

ENERGY ARC PRINCIPLES:
- Must have shape - avoid flat energy
- Typical arc: low (intro) → medium (verse) → high (chorus) → low (bridge) → peak (final chorus) → fade (outro)
- Contrast between sections (A vs B should differ meaningfully)

KEY CHANGES:
- Modulations should serve emotional arc
- Common: up a whole step for final chorus, down a third for bridge

OUTPUT FORMAT (JSON):
[
  {
    "name": "intro",
    "bars": [1, 8],
    "key": "Dm",
    "tempo": 72,
    "energy": "soft",
    "description": "Sparse piano entrance, establish mood"
  },
  {
    "name": "verse_A",
    "bars": [9, 24],
    "key": "Dm",
    "tempo": 72,
    "energy": "building",
    "description": "Add bass and drums, melody enters"
  },
  ...
]
"""

async def invoke_arranger_agent(
    style_guide: StyleGuide,
    target_length_bars: int = 80
) -> List[Section]:
    """Invoke Arranger Agent to create song structure."""
    # Similar pattern to style_agent
    # Returns list of Section objects
    pass
```

#### 2.3 Orchestrator Wrapper

**File:** `backend/app/services/agents/orchestrator.py` (new)

```python
from app.services.mir.schema import StyleGuide, Section, ChordProgression
from app.services.agents.style_agent import invoke_style_agent
from app.services.agents.arranger_agent import invoke_arranger_agent
from app.services.agents.harmony_agent import invoke_harmony_agent
from app.services.mir.compiler import compile_progression_to_tool_calls
from typing import List, Dict

async def orchestrate_composition(
    user_prompt: str,
    target_bars: int = 32
) -> Dict:
    """
    Orchestrate full composition through subagents.

    Returns compiled tool calls ready for frontend execution.
    """
    # Step 1: Style Agent
    style_guide = await invoke_style_agent(user_prompt)

    # Step 2: Arranger Agent
    sections = await invoke_arranger_agent(style_guide, target_bars)

    # Step 3: Harmony Agent (for each section)
    all_tool_calls = []
    track_id = 1  # Will be set dynamically later

    for section in sections:
        progression = await invoke_harmony_agent(style_guide, section, "piano")
        tool_calls = compile_progression_to_tool_calls(progression, track_id)
        all_tool_calls.extend(tool_calls)

    return {
        "style_guide": style_guide,
        "sections": sections,
        "tool_calls": all_tool_calls
    }
```

#### 2.4 New Hybrid Agent Tool

**File:** `backend/app/services/hybrid_agent.py` (modify)

```python
from app.services.agents.orchestrator import orchestrate_composition

@tool
def generateComposition(
    description: str,
    bars: int = 32,
    instruments: list[str] = None
) -> str:
    """Generate a complete musical composition with intelligent structure and harmony.

    Uses multi-agent system to create:
    - Style-consistent arrangement
    - Proper song structure (intro/verse/chorus/outro)
    - Voice-led chord progressions
    - Multiple instruments working together

    Args:
        description: Style description (e.g., "jazz ballad in Dm", "upbeat funk")
        bars: Total length in bars (16, 32, 64)
        instruments: List of instruments (e.g., ["piano", "bass", "drums"])

    Returns:
        JSON with track creation and note generation
    """
    return '{"status": "pending_orchestration"}'

# Add processing in process_tool_calls
if tc["name"] == "generateComposition":
    result = await orchestrate_composition(
        user_prompt=args["description"],
        target_bars=args.get("bars", 32)
    )

    # First create tracks
    track_calls = []
    for i, instrument in enumerate(args.get("instruments", ["piano"])):
        track_calls.append({
            "id": tc["id"] + f"_track_{i}",
            "name": "createTrack",
            "args": {"instrumentName": instrument}
        })

    # Then add compiled tool calls (addNotes for chords)
    processed_calls.extend(track_calls)
    processed_calls.extend(result["tool_calls"])
```

### Success Criteria

#### Automated Verification:
- [x] Style Agent tests: prompt → valid StyleGuide JSON
- [x] Arranger Agent tests: StyleGuide → valid Section list with non-overlapping bars
- [x] Orchestrator integration test: full pipeline produces tool calls
- [x] Validate energy arc has variation (not all sections same energy)

#### Manual Verification:
- [ ] User: "Create a jazz ballad in Dm, 32 bars"
  - Agent uses `generateComposition` internally
  - Creates piano track
  - Adds intro (8 bars, soft) + verse (16 bars, building) + outro (8 bars, soft)
  - All sections have consistent jazz harmony (extensions, voice leading)
  - Energy arc is shaped (not flat)
- [ ] Compare to Phase 1: Now has clear sections and energy contrast, not just chords

**Implementation Note:** After automated tests pass, manually verify the composition has clear intro/verse/outro sections with energy contrast before proceeding.

---

## Phase 3: Melody Agent & Rhythm Agent

### Overview

Add **Melody Agent** (horizontal lines, motif development) and **Rhythm Agent** (drums/percussion groove). Now compositions have structure, harmony, melody, and rhythm.

### Success Metric

**Before:** User gets harmony only, must manually add melody and drums
**After:** Full band arrangement with memorable melody and grooving rhythm section

### Changes Required

#### 3.1 Melody Agent

**File:** `backend/app/services/agents/melody_agent.py` (new)

```python
from app.services.mir.schema import StyleGuide, Section, ChordProgression, Note, MelodyPhrase
from dataclasses import dataclass
from typing import List

@dataclass
class MelodyPhrase:
    """Container for melody in a section."""
    track: str
    section: str
    notes: List[Note]
    motif_id: str = None  # For tracking recurring themes

MELODY_SYSTEM_PROMPT = """You are the Melody Agent. Create memorable melodic lines.

ROLE:
- Compose horizontal melodies that fit harmony and style
- Own contour, motif development, phrasing
- Read harmony but don't modify it

INPUT:
- StyleGuide
- Section
- ChordProgression (to know what chord tones are available)

OUTPUT: MelodyPhrase JSON

MELODIC PRINCIPLES:
1. Strong beats: chord tones. Weak beats: passing tones, tensions
2. Stepwise motion is singable; leaps create tension
3. Motif recurrence creates coherence - repeat and vary
4. Melodic peak aligns with section climax
5. Leave space - rests are part of the melody
6. Range: C4-G5 for singable melodies

PHRASING:
- 2-4 bar phrases with rests between (singers breathe!)
- Call and response patterns
- Sequence and variation

OUTPUT FORMAT (JSON):
{
  "track": "flute",
  "section": "verse_A",
  "notes": [
    {"pitch": "D4", "bar": 1, "beat": 1.0, "duration": "quarter", "velocity": 80},
    {"pitch": "F4", "bar": 1, "beat": 2.0, "duration": "eighth", "velocity": 75},
    {"pitch": "E4", "bar": 1, "beat": 2.5, "duration": "eighth", "velocity": 70},
    ...
  ],
  "motif_id": "motif_A"
}
"""

async def invoke_melody_agent(
    style_guide: StyleGuide,
    section: Section,
    harmony: ChordProgression
) -> MelodyPhrase:
    """Generate melody that fits harmony."""
    # Implementation similar to harmony_agent
    pass
```

#### 3.2 Rhythm Agent

**File:** `backend/app/services/agents/rhythm_agent.py` (new)

```python
from app.services.mir.schema import StyleGuide, Section, DrumPattern
from dataclasses import dataclass
from typing import List

@dataclass
class DrumHit:
    """Single drum hit."""
    instrument: str  # "kick", "snare", "hihat_closed", "crash"
    bar: int
    beat: float
    velocity: int

@dataclass
class DrumPattern:
    """Drum groove for a section."""
    track: str
    section: str
    hits: List[DrumHit]
    swing: float
    variation_every_n_bars: int = 4

RHYTHM_SYSTEM_PROMPT = """You are the Rhythm Agent. Create grooves that support the composition.

ROLE:
- Own time feel and pocket
- Drums/percussion only
- Set the groove that other instruments lock to

INPUT:
- StyleGuide (swing amount!)
- Section (energy level)

OUTPUT: DrumPattern JSON

RHYTHM PRINCIPLES:
1. Kick and bass align on downbeats
2. Hi-hat/ride carries subdivision feel
3. Snare defines backbeat (beats 2 and 4 in 4/4)
4. Less is more - busy drums obscure melody
5. Fills set up the next section (not show off)
6. Ghost notes add feel without clutter

PATTERN STRATEGY:
- Core pattern repeats (don't specify every bar)
- Variations every 4 or 8 bars
- Fills at section boundaries

DRUM MAPPING:
- kick: 36
- snare: 38
- hihat_closed: 42
- hihat_open: 46
- crash: 49
- ride: 51

OUTPUT FORMAT (JSON):
{
  "track": "drums",
  "section": "verse_A",
  "swing": 0.55,
  "variation_every_n_bars": 4,
  "hits": [
    {"instrument": "kick", "bar": 1, "beat": 1.0, "velocity": 100},
    {"instrument": "snare", "bar": 1, "beat": 2.0, "velocity": 90},
    {"instrument": "hihat_closed", "bar": 1, "beat": 1.0, "velocity": 70},
    {"instrument": "hihat_closed", "bar": 1, "beat": 1.5, "velocity": 60},
    ...
  ]
}
"""
```

#### 3.3 Compiler Extensions

**File:** `backend/app/services/mir/compiler.py` (modify)

```python
def compile_melody_to_notes(phrase: MelodyPhrase) -> List[Dict]:
    """Compile MelodyPhrase → addNotes format."""
    notes = []
    for note in phrase.notes:
        tick = beats_to_ticks(note.bar, note.beat)
        duration_ticks = duration_to_ticks(note.duration)
        midi_pitch = pitch_string_to_midi(note.pitch)

        notes.append({
            "pitch": midi_pitch,
            "start": tick,
            "duration": duration_ticks,
            "velocity": note.velocity
        })

    return notes

def compile_drums_to_notes(pattern: DrumPattern) -> List[Dict]:
    """Compile DrumPattern → addNotes format."""
    notes = []
    for hit in pattern.hits:
        tick = beats_to_ticks(hit.bar, hit.beat)
        # Apply swing offset to off-beats
        if pattern.swing > 0 and (hit.beat % 1) == 0.5:
            swing_offset = int(pattern.swing * 240)  # Swing eighth notes
            tick += swing_offset

        # Map drum instrument name to MIDI note
        drum_map = {
            "kick": 36, "snare": 38, "hihat_closed": 42,
            "hihat_open": 46, "crash": 49, "ride": 51
        }
        midi_pitch = drum_map.get(hit.instrument, 38)

        notes.append({
            "pitch": midi_pitch,
            "start": tick,
            "duration": 120,  # Drums typically short
            "velocity": hit.velocity
        })

    return notes
```

#### 3.4 Orchestrator Update

**File:** `backend/app/services/agents/orchestrator.py` (modify)

```python
from app.services.agents.melody_agent import invoke_melody_agent
from app.services.agents.rhythm_agent import invoke_rhythm_agent
from app.services.mir.compiler import compile_melody_to_notes, compile_drums_to_notes

async def orchestrate_composition(
    user_prompt: str,
    target_bars: int = 32
) -> Dict:
    """Full composition with harmony, melody, and rhythm."""

    # Steps 1-3: Style, Arranger, Harmony (from Phase 2)
    style_guide = await invoke_style_agent(user_prompt)
    sections = await invoke_arranger_agent(style_guide, target_bars)

    # Track IDs
    piano_track_id = 1
    melody_track_id = 2
    drums_track_id = 3

    all_tool_calls = []

    # Create tracks
    all_tool_calls.append({
        "name": "createTrack",
        "args": {"instrumentName": "piano", "trackName": "Piano"}
    })
    all_tool_calls.append({
        "name": "createTrack",
        "args": {"instrumentName": "flute", "trackName": "Melody"}
    })
    all_tool_calls.append({
        "name": "createTrack",
        "args": {"instrumentName": "drums", "trackName": "Drums"}
    })

    for section in sections:
        # Harmony
        harmony = await invoke_harmony_agent(style_guide, section, "piano")
        harmony_calls = compile_progression_to_tool_calls(harmony, piano_track_id)
        all_tool_calls.extend(harmony_calls)

        # Melody (only in verse/chorus, not intro/outro)
        if section.name not in ["intro", "outro"]:
            melody = await invoke_melody_agent(style_guide, section, harmony)
            melody_notes = compile_melody_to_notes(melody)
            all_tool_calls.append({
                "name": "addNotes",
                "args": {"trackId": melody_track_id, "notes": melody_notes}
            })

        # Rhythm
        rhythm = await invoke_rhythm_agent(style_guide, section)
        drum_notes = compile_drums_to_notes(rhythm)
        all_tool_calls.append({
            "name": "addNotes",
            "args": {"trackId": drums_track_id, "notes": drum_notes}
        })

    return {
        "style_guide": style_guide,
        "sections": sections,
        "tool_calls": all_tool_calls
    }
```

### Success Criteria

#### Automated Verification:
- [x] Melody Agent generates notes in valid range (C4-G5)
- [x] Melody notes are chord tones on strong beats (validate against harmony)
- [x] Rhythm Agent outputs valid drum MIDI note numbers (36-51)
- [x] Swing offset applied correctly to off-beat notes
- [x] Full orchestration produces 3 tracks (piano, melody, drums)

#### Manual Verification:
- [ ] User: "Create a jazz ballad in Dm, 32 bars"
  - Gets 3 tracks: Piano (chords), Melody (flute), Drums
  - Melody is singable and fits over the chords
  - Drums have swing feel matching style
  - Melody has motif recurrence (phrase A repeated with variation)
  - Drums are sparse in intro, fuller in verse
- [ ] Compare to Phase 2: Now has melody and rhythm, feels like a complete song

**Implementation Note:** After tests pass, manually play the composition and verify melody is memorable and drums groove properly before proceeding.

---

## Phase 4: Critic Agent & Revision Loop

### Overview

Add **Critic Agent** to evaluate composition quality and trigger targeted revisions. This is the quality control layer that ensures musical correctness before compilation.

### Success Metric

**Before:** Generated music may have voice leading errors, range violations, or style inconsistencies
**After:** Critic catches errors, sends sections back for revision until quality threshold met

### Changes Required

#### 4.1 MIR Validators (Rule-based)

**File:** `backend/app/services/mir/validators.py` (new)

```python
from app.services.mir.schema import Chord, ChordProgression, MelodyPhrase, StyleGuide
from typing import List, Dict

def validate_voice_leading(progression: ChordProgression) -> List[Dict]:
    """Check for voice leading errors (parallel fifths, excessive jumps)."""
    errors = []

    for i in range(len(progression.chords) - 1):
        curr_chord = progression.chords[i]
        next_chord = progression.chords[i + 1]

        # Convert voicings to MIDI numbers
        curr_pitches = [pitch_string_to_midi(p) for p in curr_chord.voicing]
        next_pitches = [pitch_string_to_midi(p) for p in next_chord.voicing]

        # Check for parallel fifths/octaves
        for voice_idx in range(min(len(curr_pitches), len(next_pitches))):
            interval_curr = (curr_pitches[voice_idx] - curr_pitches[0]) % 12
            interval_next = (next_pitches[voice_idx] - next_pitches[0]) % 12

            if interval_curr in [0, 7] and interval_curr == interval_next:
                # Parallel fifth or octave
                errors.append({
                    "type": "parallel_fifth",
                    "severity": "error",
                    "location": f"bar {curr_chord.bar}",
                    "message": f"Parallel {interval_curr}-semitone interval in voice {voice_idx}"
                })

        # Check for excessive jumps (>5 semitones in any voice)
        for voice_idx in range(min(len(curr_pitches), len(next_pitches))):
            jump = abs(next_pitches[voice_idx] - curr_pitches[voice_idx])
            if jump > 5:
                errors.append({
                    "type": "large_jump",
                    "severity": "warning",
                    "location": f"bar {curr_chord.bar}",
                    "message": f"Voice {voice_idx} jumps {jump} semitones"
                })

    return errors

def validate_melody_range(phrase: MelodyPhrase, max_pitch: str = "G5", min_pitch: str = "C4") -> List[Dict]:
    """Check melody stays in singable range."""
    errors = []
    max_midi = pitch_string_to_midi(max_pitch)
    min_midi = pitch_string_to_midi(min_pitch)

    for note in phrase.notes:
        midi_pitch = pitch_string_to_midi(note.pitch)
        if midi_pitch > max_midi:
            errors.append({
                "type": "range_violation",
                "severity": "error",
                "location": f"bar {note.bar}",
                "message": f"Note {note.pitch} exceeds max range {max_pitch}"
            })
        elif midi_pitch < min_midi:
            errors.append({
                "type": "range_violation",
                "severity": "error",
                "location": f"bar {note.bar}",
                "message": f"Note {note.pitch} below min range {min_pitch}"
            })

    return errors

def validate_style_consistency(
    harmony: ChordProgression,
    style_guide: StyleGuide
) -> List[Dict]:
    """Check if harmony uses only allowed extensions."""
    errors = []

    for chord in harmony.chords:
        # Parse quality to extract extensions (e.g., "m9" → ["9"])
        extensions = []
        if "9" in chord.quality:
            extensions.append("9")
        if "11" in chord.quality:
            extensions.append("11")
        if "13" in chord.quality:
            extensions.append("13")

        for ext in extensions:
            if ext not in style_guide.extensions_allowed:
                errors.append({
                    "type": "style_violation",
                    "severity": "warning",
                    "location": f"bar {chord.bar}",
                    "message": f"Extension '{ext}' not allowed in {style_guide.genre} style"
                })

    return errors
```

#### 4.2 Critic Agent

**File:** `backend/app/services/agents/critic_agent.py` (new)

```python
from app.services.mir.schema import ChordProgression, MelodyPhrase, StyleGuide
from app.services.mir.validators import (
    validate_voice_leading,
    validate_melody_range,
    validate_style_consistency
)
from dataclasses import dataclass
from typing import List, Dict
import json

@dataclass
class CriticReport:
    """Critic evaluation report."""
    overall_score: float  # 0.0-1.0
    issues: List[Dict]  # List of errors/warnings
    passed: bool
    revision_needed: List[str]  # Which agents to re-run: ["harmony", "melody"]

CRITIC_SYSTEM_PROMPT = """You are the Critic Agent. Evaluate composition quality.

ROLE:
- Judge musical correctness and style consistency
- Authority to send work back for revision
- Only agent that doesn't create - only evaluates

EVALUATION DIMENSIONS:
1. Harmonic: Voice leading, progression logic, style fit
2. Melodic: Chord/scale fit, contour quality, motif development
3. Rhythmic: Groove quality, variation, melody support
4. Stylistic: Consistency with StyleGuide across all elements
5. Technical: Range violations, playability, MIDI validity

SCORING:
- 1.0 = Perfect
- 0.8-0.99 = Good, minor issues
- 0.6-0.79 = Acceptable, some revision recommended
- < 0.6 = Needs revision

REVISION THRESHOLD:
- score >= 0.8 AND no errors → proceed to compilation
- else → route issues to responsible agents → re-evaluate (max 3 cycles)

OUTPUT FORMAT (JSON):
{
  "overall_score": 0.85,
  "issues": [
    {
      "type": "parallel_fifth",
      "agent": "harmony",
      "severity": "error",
      "location": "bar 5",
      "message": "Parallel fifths between chords",
      "suggestion": "Use contrary motion in inner voices"
    }
  ],
  "passed": false,
  "revision_needed": ["harmony"]
}
"""

async def invoke_critic_agent(
    style_guide: StyleGuide,
    harmony: ChordProgression,
    melody: MelodyPhrase = None,
    rhythm = None
) -> CriticReport:
    """Evaluate composition and return critique."""

    all_issues = []

    # Run rule-based validators first
    all_issues.extend(validate_voice_leading(harmony))
    all_issues.extend(validate_style_consistency(harmony, style_guide))

    if melody:
        all_issues.extend(validate_melody_range(melody))

    # Count errors vs warnings
    error_count = sum(1 for issue in all_issues if issue["severity"] == "error")
    warning_count = sum(1 for issue in all_issues if issue["severity"] == "warning")

    # Calculate score
    # Start at 1.0, deduct 0.2 per error, 0.05 per warning
    score = max(0.0, 1.0 - (error_count * 0.2) - (warning_count * 0.05))

    # Determine if revision needed
    passed = score >= 0.8 and error_count == 0

    # Group issues by responsible agent
    revision_needed = []
    if any(i["type"] in ["parallel_fifth", "large_jump", "style_violation"] for i in all_issues):
        revision_needed.append("harmony")
    if any(i["type"] == "range_violation" for i in all_issues):
        revision_needed.append("melody")

    return CriticReport(
        overall_score=score,
        issues=all_issues,
        passed=passed,
        revision_needed=list(set(revision_needed))  # Unique
    )
```

#### 4.3 Revision Loop in Orchestrator

**File:** `backend/app/services/agents/orchestrator.py` (modify)

```python
from app.services.agents.critic_agent import invoke_critic_agent

async def orchestrate_composition_with_revision(
    user_prompt: str,
    target_bars: int = 32,
    max_revision_cycles: int = 3
) -> Dict:
    """Orchestrate with quality validation and revision."""

    # Phase 1: Style & Structure
    style_guide = await invoke_style_agent(user_prompt)
    sections = await invoke_arranger_agent(style_guide, target_bars)

    # Phase 2: Content generation with revision loop
    revision_cycle = 0
    all_compositions = {}  # section_name → {harmony, melody, rhythm}

    for section in sections:
        while revision_cycle < max_revision_cycles:
            # Generate harmony
            harmony = await invoke_harmony_agent(style_guide, section, "piano")

            # Generate melody (if not intro/outro)
            melody = None
            if section.name not in ["intro", "outro"]:
                melody = await invoke_melody_agent(style_guide, section, harmony)

            # Generate rhythm
            rhythm = await invoke_rhythm_agent(style_guide, section)

            # Evaluate with Critic
            critique = await invoke_critic_agent(
                style_guide=style_guide,
                harmony=harmony,
                melody=melody,
                rhythm=rhythm
            )

            if critique.passed:
                # Quality threshold met!
                all_compositions[section.name] = {
                    "harmony": harmony,
                    "melody": melody,
                    "rhythm": rhythm
                }
                break
            else:
                # Log issues and retry
                print(f"[Critic] Section {section.name} failed (score: {critique.overall_score})")
                print(f"[Critic] Issues: {critique.issues}")
                print(f"[Critic] Revising: {critique.revision_needed}")
                revision_cycle += 1

                # In production: provide critique feedback to agents for targeted revision
                # For now: regenerate from scratch

        if revision_cycle >= max_revision_cycles:
            # Exceeded max cycles - accept best effort
            print(f"[Critic] WARNING: Section {section.name} did not pass after {max_revision_cycles} cycles")
            all_compositions[section.name] = {
                "harmony": harmony,
                "melody": melody,
                "rhythm": rhythm
            }

    # Phase 3: Compile to tool calls
    all_tool_calls = []

    # Create tracks
    all_tool_calls.append({"name": "createTrack", "args": {"instrumentName": "piano"}})
    all_tool_calls.append({"name": "createTrack", "args": {"instrumentName": "flute"}})
    all_tool_calls.append({"name": "createTrack", "args": {"instrumentName": "drums"}})

    piano_track_id = 1
    melody_track_id = 2
    drums_track_id = 3

    for section_name, content in all_compositions.items():
        # Compile harmony
        harmony_calls = compile_progression_to_tool_calls(content["harmony"], piano_track_id)
        all_tool_calls.extend(harmony_calls)

        # Compile melody
        if content["melody"]:
            melody_notes = compile_melody_to_notes(content["melody"])
            all_tool_calls.append({
                "name": "addNotes",
                "args": {"trackId": melody_track_id, "notes": melody_notes}
            })

        # Compile rhythm
        drum_notes = compile_drums_to_notes(content["rhythm"])
        all_tool_calls.append({
            "name": "addNotes",
            "args": {"trackId": drums_track_id, "notes": drum_notes}
        })

    return {
        "style_guide": style_guide,
        "sections": sections,
        "tool_calls": all_tool_calls,
        "revision_cycles": revision_cycle
    }
```

### Success Criteria

#### Automated Verification:
- [x] Voice leading validator catches parallel fifths: `pytest backend/app/services/mir/test_validators.py`
- [x] Range validator catches out-of-range notes
- [x] Style validator catches forbidden extensions
- [x] Critic agent returns CriticReport with score and issues
- [x] Revision loop terminates (max 3 cycles) even if never passes

#### Manual Verification:
- [ ] Inject a parallel fifth into test harmony → Critic catches it with error
- [ ] Inject out-of-range note (A5) in melody → Critic catches it
- [ ] User: "Create a pop song" (simple harmony style)
  - If harmony uses complex jazz extensions (13ths), Critic flags style violation
  - Harmony agent regenerates with simpler chords
  - Final composition passes critique
- [ ] Compare to Phase 3: Generated music now has fewer musical errors

**Implementation Note:** After tests pass, manually test that the revision loop actually improves quality (inject intentional errors and verify they're caught and fixed).

---

## Phase 5: Bass & Full Orchestration

### Overview

Final phase adds **Bass Agent** (walking bass, root motion) and **Orchestration Agent** (assigns parts to instruments, dynamics, register management). This completes the multi-agent system.

### Success Metric

**Before:** Fixed instrumentation (piano, melody, drums), no bass, no dynamic variation
**After:** Full band arrangements with bass, flexible instrumentation, dynamic contour

### Changes Required

#### 5.1 Bass Agent

**File:** `backend/app/services/agents/bass_agent.py` (new)

```python
from app.services.mir.schema import ChordProgression, Section, StyleGuide, Note
from dataclasses import dataclass
from typing import List

@dataclass
class BassLine:
    """Bass line for a section."""
    track: str
    section: str
    notes: List[Note]

BASS_SYSTEM_PROMPT = """You are the Bass Agent. Create bass lines that anchor the harmony.

ROLE:
- Generate bass lines that outline chord progressions
- Lock rhythmically with drums
- Own the low register

INPUT:
- ChordProgression (to extract root notes and function)
- StyleGuide (straight vs swing, energy)
- Section (tempo, energy)

OUTPUT: BassLine JSON

BASS PRINCIPLES:
1. Root notes on beat 1 (most important)
2. Approach notes (chromatic or scale-wise) lead to next chord
3. Octave jumps add energy
4. Walking bass (jazz): stepwise motion through chord tones and passing tones
5. Locked to kick drum pattern rhythmically
6. Range: E1-E3 (octaves 1-3)

PATTERNS BY STYLE:
- Jazz: walking bass (quarter notes), chromatic approach
- Rock: root-fifth-root, eighth notes, some syncopation
- Pop: simple root notes on beats 1 and 3, occasional fills
- Funk: syncopated sixteenths, dead notes (ghosted)

OUTPUT FORMAT (JSON):
{
  "track": "bass",
  "section": "verse_A",
  "notes": [
    {"pitch": "D2", "bar": 1, "beat": 1.0, "duration": "quarter", "velocity": 90},
    {"pitch": "E2", "bar": 1, "beat": 2.0, "duration": "quarter", "velocity": 85},
    {"pitch": "F2", "bar": 1, "beat": 3.0, "duration": "quarter", "velocity": 85},
    {"pitch": "A2", "bar": 1, "beat": 4.0, "duration": "quarter", "velocity": 80},
    ...
  ]
}
"""

async def invoke_bass_agent(
    style_guide: StyleGuide,
    section: Section,
    harmony: ChordProgression
) -> BassLine:
    """Generate bass line from chord progression."""
    # Similar pattern to other agents
    pass
```

#### 5.2 Orchestration Agent

**File:** `backend/app/services/agents/orchestration_agent.py` (new)

```python
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class InstrumentAssignment:
    """Maps MIR content to specific instrument."""
    source_content: str  # "harmony", "melody", "bass", "rhythm"
    instrument: str  # "piano", "guitar", "flute", "electric_bass"
    register_shift: int = 0  # Octaves to shift up/down
    dynamic_envelope: List[tuple[int, int]] = None  # [(bar, velocity), ...]

@dataclass
class OrchestrationPlan:
    """Complete orchestration specification."""
    assignments: List[InstrumentAssignment]
    track_order: List[str]  # Order tracks should appear
    dynamic_arc: Dict[str, List[tuple[int, int]]]  # section → [(bar, overall_velocity)]

ORCHESTRATION_SYSTEM_PROMPT = """You are the Orchestration Agent. Assign parts to instruments.

ROLE:
- Decide who plays what, when they enter/exit
- Register adjustments (move bass line down, melody up)
- Dynamic envelopes per section
- Last creative step before critique

INPUT:
- StyleGuide (genre informs instrument choices)
- All MIR content (harmony, melody, bass, rhythm)
- Section list (energy arc)

OUTPUT: OrchestrationPlan JSON

ORCHESTRATION PRINCIPLES:
1. Start sparse, build density toward climax
2. Each instrument has a clear role
3. Register separation prevents mud (bass low, melody high)
4. Dynamics should breathe - not everything at max
5. Use silence as an orchestration tool

INSTRUMENT ROLES BY STYLE:
- Jazz: piano (chords), bass (walking), drums (swing), sax/trumpet (melody)
- Rock: guitar (power chords or arpeggios), bass (root-fifth), drums (backbeat), keys (pads)
- Pop: keys (chords), bass (simple root), drums (straightforward), synth/vocal melody
- Classical: strings (harmony), woodwinds (melody), brass (accents), timpani (rhythm)

ENTRY/EXIT:
- Intro: 1-2 instruments establish mood
- Verse: Add melody, keep sparse
- Chorus: Full band, highest density
- Bridge: Change texture (maybe drop drums, feature piano solo)
- Outro: Gradually remove instruments

OUTPUT FORMAT (JSON):
{
  "assignments": [
    {
      "source_content": "harmony",
      "instrument": "piano",
      "register_shift": 0,
      "active_sections": ["intro", "verse_A", "chorus_B", "outro"]
    },
    {
      "source_content": "melody",
      "instrument": "flute",
      "register_shift": 1,
      "active_sections": ["verse_A", "chorus_B"]
    },
    ...
  ],
  "track_order": ["drums", "bass", "piano", "flute"],
  "dynamic_arc": {
    "intro": [(1, 60), (8, 65)],
    "verse_A": [(9, 70), (24, 80)],
    "chorus_B": [(25, 90), (40, 95)]
  }
}
"""

async def invoke_orchestration_agent(
    style_guide: StyleGuide,
    sections: List[Section],
    all_content: Dict  # {section_name: {harmony, melody, bass, rhythm}}
) -> OrchestrationPlan:
    """Create orchestration plan."""
    # Returns InstrumentAssignment list
    pass
```

#### 5.3 Final Orchestrator

**File:** `backend/app/services/agents/orchestrator.py` (final version)

```python
from app.services.agents.bass_agent import invoke_bass_agent
from app.services.agents.orchestration_agent import invoke_orchestration_agent

async def orchestrate_full_composition(
    user_prompt: str,
    target_bars: int = 32
) -> Dict:
    """Complete multi-agent composition pipeline."""

    # Phase 1: Planning
    style_guide = await invoke_style_agent(user_prompt)
    sections = await invoke_arranger_agent(style_guide, target_bars)

    # Phase 2: Content generation per section (with revision)
    all_content = {}

    for section in sections:
        revision_cycle = 0
        while revision_cycle < 3:
            # Generate all content
            harmony = await invoke_harmony_agent(style_guide, section, "piano")

            melody = None
            if section.name not in ["intro", "outro"]:
                melody = await invoke_melody_agent(style_guide, section, harmony)

            bass = await invoke_bass_agent(style_guide, section, harmony)
            rhythm = await invoke_rhythm_agent(style_guide, section)

            # Critique
            critique = await invoke_critic_agent(
                style_guide, harmony, melody, rhythm
            )

            if critique.passed:
                all_content[section.name] = {
                    "harmony": harmony,
                    "melody": melody,
                    "bass": bass,
                    "rhythm": rhythm
                }
                break

            revision_cycle += 1

    # Phase 3: Orchestration (assigns to instruments, dynamics)
    orchestration = await invoke_orchestration_agent(style_guide, sections, all_content)

    # Phase 4: Compilation to MIDI tool calls
    all_tool_calls = []
    track_map = {}  # instrument_name → track_id

    # Create tracks based on orchestration
    for idx, assignment in enumerate(orchestration.assignments):
        instrument = assignment.instrument
        if instrument not in track_map:
            track_id = len(track_map) + 1
            track_map[instrument] = track_id
            all_tool_calls.append({
                "name": "createTrack",
                "args": {"instrumentName": instrument}
            })

    # Compile content with orchestration
    for section_name, content in all_content.items():
        section_obj = next(s for s in sections if s.name == section_name)

        # Find which instruments play each part in this section
        for assignment in orchestration.assignments:
            if section_name not in assignment.get("active_sections", [section_name]):
                continue  # This instrument doesn't play in this section

            track_id = track_map[assignment.instrument]

            # Get the content
            if assignment.source_content == "harmony":
                mir_content = content["harmony"]
                notes = compile_progression_to_tool_calls(mir_content, track_id)
            elif assignment.source_content == "melody":
                mir_content = content["melody"]
                if mir_content:
                    notes = compile_melody_to_notes(mir_content)
                    notes = [{"name": "addNotes", "args": {"trackId": track_id, "notes": notes}}]
                else:
                    continue
            elif assignment.source_content == "bass":
                mir_content = content["bass"]
                notes = compile_melody_to_notes(mir_content)  # Bass uses same format as melody
                notes = [{"name": "addNotes", "args": {"trackId": track_id, "notes": notes}}]
            elif assignment.source_content == "rhythm":
                mir_content = content["rhythm"]
                notes = compile_drums_to_notes(mir_content)
                notes = [{"name": "addNotes", "args": {"trackId": track_id, "notes": notes}}]

            all_tool_calls.extend(notes)

    return {
        "style_guide": style_guide,
        "sections": sections,
        "orchestration": orchestration,
        "tool_calls": all_tool_calls
    }
```

### Success Criteria

#### Automated Verification:
- [x] Bass Agent generates notes in bass range (E1-E3)
- [x] Bass notes align with chord roots on beat 1
- [x] Orchestration Agent assigns all content to instruments
- [x] No instrument has empty assignment
- [x] Dynamic arc shows variation (not all sections same velocity)

#### Manual Verification:
- [ ] User: "Create a jazz ballad in Dm, 64 bars"
  - Gets 4+ tracks: bass, piano, drums, melody (e.g., sax)
  - Bass locks with kick drum on downbeats
  - Bass plays walking patterns (quarter notes, chromatic approach)
  - Intro has 1-2 instruments only
  - Chorus has all instruments at higher dynamics
  - Outro fades with instruments dropping out
- [ ] Compare to Phase 4: Now has bass line and dynamic variation across sections

**Implementation Note:** This is the final phase. After tests pass, perform full end-to-end test: user prompt → complete multi-agent pipeline → playable song with all elements.

---

## Performance Considerations

### Latency

**Current single-agent:** ~3-5 seconds per composition (1 LLM call)
**Multi-agent Phase 5:** ~15-30 seconds per composition (7+ LLM calls in sequence)

**Mitigation:**
- Use streaming endpoints (`/api/agent/step/stream`) to show progress
- Cache StyleGuide for repeated compositions in same style
- Consider parallel agent calls where possible (harmony + melody + rhythm for same section)

### Cost

**Estimate per composition (Claude Sonnet via OpenRouter):**
- Style Agent: ~500 tokens → $0.003
- Arranger: ~800 tokens → $0.005
- Harmony Agent (per section): ~1000 tokens × 4 sections → $0.024
- Melody Agent (per section): ~1200 tokens × 3 sections → $0.022
- Rhythm Agent (per section): ~800 tokens × 4 sections → $0.019
- Bass Agent (per section): ~1000 tokens × 4 sections → $0.024
- Orchestration: ~1500 tokens → $0.009
- Critic (per section): ~600 tokens × 4 sections → $0.014

**Total per composition: ~$0.12** (vs ~$0.02 for current single-agent)

**Mitigation:**
- Use GPT-4o-mini for simpler agents (Style, Arranger) → 10x cheaper
- Only run Critic on final output, not intermediate revisions
- User controls composition length (fewer bars = fewer sections = lower cost)

---

## Migration Notes

### Backward Compatibility

The existing 17 MIDI tools remain unchanged. Users can still:
- Chat directly for simple edits: "transpose this up an octave"
- Use individual tools: `addNotes`, `setTempo`
- Mix old and new approaches: generate with multi-agent, edit with simple tools

### Data Model

No changes to:
- `Song` class (MobX)
- `Track` structure
- MIDI event format
- Frontend `toolExecutor.ts`

All MIR schemas exist only in backend, compiled away before reaching frontend.

### Rollout Strategy

Each phase can be deployed independently:
1. **Phase 1:** Users get `addChordProgression` tool (better harmony)
2. **Phase 2:** `generateComposition` appears (structure + harmony)
3. **Phase 3:** `generateComposition` improves (adds melody + drums)
4. **Phase 4:** Quality improves (fewer errors)
5. **Phase 5:** Full arrangements (bass + orchestration)

At any point, can pause rollout and continue with current functionality.

---

## Testing Strategy

### Unit Tests

Each phase adds:
- MIR schema serialization tests
- Compiler correctness tests (MIR → MIDI math)
- Agent prompt→output format validation
- Validator logic tests (voice leading, range, style)

**Target:** 80%+ code coverage on `backend/app/services/mir/` and `backend/app/services/agents/`

### Integration Tests

- Full pipeline tests: user prompt → tool calls → verify tool call structure
- Song state tests: execute tool calls → verify Song object state matches expected
- Round-trip tests: MIR → compile → execute → export MIDI → re-import → verify notes match

### Musical Quality Tests (Automated where possible)

- Voice leading checker: no parallel fifths in generated progressions
- Range checker: all melodies in C4-G5
- Harmonic rhythm checker: changes match energy level
- Style consistency: extensions match genre

### Manual QA Checklist (per phase)

- [ ] Generate composition in multiple styles (jazz, rock, pop, classical)
- [ ] Verify each style sounds appropriate
- [ ] Check for musical errors (parallel fifths, out-of-range notes)
- [ ] Verify revision loop catches and fixes errors
- [ ] Test edge cases (very short song, very long song, unusual key)
- [ ] Performance test: measure end-to-end latency

---

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph Multi-Agent Systems](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/)
- [Music Theory: Voice Leading](https://www.musictheory.net/lessons/51)
- [General MIDI Specification](https://www.midi.org/specifications-old/item/gm-level-1-sound-set)
- [MobX Reactivity](https://mobx.js.org/)
- [DeepAgents Library](https://github.com/langchain-ai/deepagents)

---

## Summary Timeline

| Phase | Focus | Duration Estimate | Cumulative Quality Gain |
|-------|-------|-------------------|------------------------|
| **Phase 1** | MIR + Harmony Agent | 1.5 weeks | Better voice leading |
| **Phase 2** | Style + Arranger | 1.5 weeks | Song structure |
| **Phase 3** | Melody + Rhythm | 2 weeks | Full band feel |
| **Phase 4** | Critic + Revision | 1.5 weeks | Fewer errors |
| **Phase 5** | Bass + Orchestration | 2 weeks | Professional arrangements |
| **Total** | | **8.5 weeks** | Production-quality compositions |

**Each phase delivers testable improvements while preserving the existing chat UX.**

---

*End of Implementation Plan*
