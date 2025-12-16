"""Musical Content Agent - Unified agent for harmony, melody, and bass.

This agent intelligently generates harmony, melody, and bass together,
considering their relationships:
- Harmony provides chord progressions and harmonic context
- Melody uses chord tones and creates melodic interest
- Bass outlines harmony and locks with rhythm
- All elements work together for musical coherence
"""

from app.services.mir.schema import (
    StyleGuide, Section, ChordProgression, Chord, MelodyPhrase, Note, BassLine
)
from langchain_openai import ChatOpenAI
from app.config import get_settings
from typing import Dict, Optional
import json
import re

settings = get_settings()

MUSICAL_CONTENT_SYSTEM_PROMPT = """You are the Musical Content Agent, a specialist in creating harmonically coherent musical content.

ROLE:
- Generate harmony (chord progressions), melody, and bass together as a unified musical statement
- Consider how all elements interact: melody uses chord tones, bass outlines harmony, rhythm ties everything together
- Create suspense, resolution, and musical flow through harmonic and melodic choices
- Ensure all parts work together rhythmically and harmonically

INPUT:
- StyleGuide: genre, harmonic complexity, tempo, swing feel
- Section: bars, key, tempo, energy level
- Optional: existing rhythm pattern (for timing reference)

OUTPUT:
- Unified JSON containing harmony, melody, and bass that work together

MUSICAL PRINCIPLES:

HARMONY (Chord Progressions):
1. Strong beats get stable chords (I, IV, V)
2. Weak beats allow passing chords and suspensions
3. Voice leading: minimize movement, stepwise preferred
4. Match complexity to style (jazz: 9ths/11ths/13ths, pop: simpler)
5. Harmonic rhythm matches tempo (fast = slower changes, slow = more changes)
6. Cadences match section function (half cadence for question, authentic for answer)
7. Create tension and resolution through chord progressions
8. Use suspensions and resolutions for musical interest

MELODY (Melodic Lines):
1. Strong beats: chord tones. Weak beats: passing tones, tensions
2. Stepwise motion is singable; leaps create tension and interest
3. Motif recurrence creates coherence - repeat and vary
4. Melodic peak aligns with section climax
5. Leave space - rests are part of the melody
6. Range: C4-G5 for singable melodies
7. Use chord tones on strong beats (beat 1.0, 3.0)
8. Create melodic interest through contour, rhythm, and phrasing
9. Build suspense through ascending lines, resolve through descending

BASS (Bass Lines):
1. Root notes on beat 1 are CRITICAL - always anchor each chord
2. Approach notes (chromatic or scale-wise) lead smoothly to next chord
3. Octave jumps add energy (use sparingly)
4. Walking bass (jazz): stepwise motion through chord tones and passing tones
5. Lock rhythmically with kick drum pattern - downbeats especially
6. Range: E1-E3 (octaves 1-3) - stay in bass register
7. Use chord tones (root, third, fifth, seventh) on strong beats
8. Use passing tones (chromatic or scalar) on weak beats
9. Create groove through rhythmic patterns

INTEGRATION (How Parts Work Together):
1. Melody should use notes from the harmony chords (chord tones)
2. Bass should outline the harmony (root on beat 1 of each chord)
3. All parts should align rhythmically (strong beats together)
4. Create call-and-response between melody and bass
5. Use counterpoint: when melody goes up, bass can go down (and vice versa)
6. Space: when melody is active, bass can be simpler (and vice versa)
7. Build energy: all parts can increase density together in chorus
8. Create contrast: sparse sections vs dense sections

TIMING AND SPACING:
1. Strong beats (1, 3): all parts align
2. Weak beats (2, 4): can have passing tones, syncopation
3. Subdivision: eighth notes for groove, sixteenths for energy
4. Rests are musical - use them strategically
5. Syncopation: place notes between beats for groove
6. Swing feel: delay off-beat notes by swing amount

GROOVE:
1. Lock bass and kick drum on downbeats
2. Melody can float above with syncopation
3. Harmony provides rhythmic comping (chord stabs on beats)
4. All parts contribute to overall groove feel
5. Vary density: sparse in verse, dense in chorus

OUTPUT FORMAT (JSON):
{
  "harmony": {
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
      }
    ]
  },
  "melody": {
    "track": "flute",
    "section": "verse_A",
    "notes": [
      {"pitch": "D4", "bar": 1, "beat": 1.0, "duration": "quarter", "velocity": 80},
      {"pitch": "F4", "bar": 1, "beat": 2.0, "duration": "eighth", "velocity": 75}
    ],
    "motif_id": "motif_A"
  },
  "bass": {
    "track": "bass",
    "section": "verse_A",
    "notes": [
      {"pitch": "D2", "bar": 1, "beat": 1.0, "duration": "quarter", "velocity": 90},
      {"pitch": "E2", "bar": 1, "beat": 2.0, "duration": "quarter", "velocity": 85}
    ]
  }
}

CRITICAL RULES:
- Return ONLY valid JSON. No explanations, no markdown, no code blocks.
- All three parts (harmony, melody, bass) must be included
- Melody notes must use chord tones from the harmony on strong beats
- Bass must start with the root of each chord on beat 1
- All parts must align rhythmically (strong beats together)
- Create musical interest through suspense, resolution, and flow
- Consider timing, spacing, and groove in all parts
"""


async def invoke_musical_content_agent(
    style_guide: StyleGuide,
    section: Section,
    harmony_track: str = "piano",
    melody_track: str = "flute",
    bass_track: str = "bass",
    rhythm_pattern: Optional[Dict] = None
) -> Dict:
    """
    Invoke the unified musical content agent to generate harmony, melody, and bass together.

    Args:
        style_guide: StyleGuide defining musical style constraints
        section: Section defining bars, key, energy
        harmony_track: Instrument name for harmony (default: "piano")
        melody_track: Instrument name for melody (default: "flute")
        bass_track: Instrument name for bass (default: "bass")
        rhythm_pattern: Optional rhythm pattern for timing reference

    Returns:
        Dictionary with:
        - harmony: ChordProgression object
        - melody: MelodyPhrase object (or None if section is intro/outro)
        - bass: BassLine object
    """
    print(f"[MUSICAL_CONTENT] Generating unified content for {section.name}...")

    model = ChatOpenAI(
        model=settings.openrouter_model,
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
        temperature=0.8,  # Higher temperature for creative musical choices
    )

    # Build rhythm context if provided
    rhythm_context = ""
    if rhythm_pattern:
        rhythm_context = f"""
Rhythm Pattern Reference:
- Kick pattern: {rhythm_pattern.get('kick_pattern', 'N/A')}
- Snare pattern: {rhythm_pattern.get('snare_pattern', 'N/A')}
- Hi-hat pattern: {rhythm_pattern.get('hihat_pattern', 'N/A')}
- Lock bass to kick drum on downbeats
"""

    # Build the user prompt
    prompt = f"""Generate unified musical content (harmony, melody, bass) for this section:

Style Guide:
- Genre: {style_guide.genre} {style_guide.subgenre}
- Harmonic Complexity: {style_guide.harmonic_complexity}
- Swing: {style_guide.swing} (0.0=straight, 0.55=ballad swing, 0.67=bebop swing)
- Tempo Range: {style_guide.tempo_range[0]}-{style_guide.tempo_range[1]} BPM
- Extensions Allowed: {', '.join(style_guide.extensions_allowed)}
- Reference Artists: {', '.join(style_guide.reference_artists) if style_guide.reference_artists else 'None'}

Section:
- Name: {section.name}
- Bars: {section.bars[0]}-{section.bars[1]} ({section.bars[1] - section.bars[0] + 1} bars)
- Key: {section.key}
- Tempo: {section.tempo} BPM
- Energy: {section.energy}
{rhythm_context}
Tracks:
- Harmony: {harmony_track}
- Melody: {melody_track}
- Bass: {bass_track}

Requirements:
1. HARMONY: Create chord progression that fits {style_guide.genre} style
   - Use {style_guide.harmonic_complexity} harmony
   - Provide explicit voicings for {harmony_track}
   - Create tension and resolution through chord choices
   - Match harmonic rhythm to tempo and energy

2. MELODY: Create melodic line that uses harmony chord tones
   - Use chord tones on strong beats (beat 1.0, 3.0)
   - Create memorable contour and phrasing
   - Range: C4-G5 for singable melodies
   - Match "{section.energy}" energy level
   - Build suspense and resolution through melodic choices
   {"- Skip melody for intro/outro sections" if section.name in ["intro", "outro"] else ""}

3. BASS: Create bass line that outlines harmony
   - Put chord root on beat 1 of each chord
   - Use appropriate style: {"walking bass" if "jazz" in style_guide.genre.lower() else "root-fifth pattern" if "rock" in style_guide.genre.lower() else "simple roots"}
   - Range: E1-E3
   - Lock rhythmically with downbeats
   - Create groove through rhythmic patterns

4. INTEGRATION: Ensure all parts work together
   - Melody uses notes from harmony chords
   - Bass outlines harmony (root on beat 1)
   - All parts align on strong beats
   - Create musical flow through the section

Return unified JSON with harmony, melody, and bass. No explanations."""

    messages = [
        {"role": "system", "content": MUSICAL_CONTENT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    response = await model.ainvoke(messages)

    # Extract JSON from response
    content = response.content.strip()

    # Remove markdown code blocks if present
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    # Remove "json" language identifier if present
    if content.startswith("json"):
        content = content[4:].strip()

    # Try to extract JSON from markdown code blocks or plain text
    json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', content, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find JSON directly
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            print(f"[MUSICAL_CONTENT] Error: Could not extract JSON from response")
            print(f"[MUSICAL_CONTENT] Raw response: {content[:500]}...")
            raise ValueError(f"Could not extract JSON from musical content agent response")

    # Parse the JSON response
    try:
        content_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"[MUSICAL_CONTENT] Error parsing JSON: {e}")
        print(f"[MUSICAL_CONTENT] JSON string: {json_str[:500]}...")
        raise ValueError(f"Invalid JSON in musical content agent response: {e}") from e

    # Convert to MIR objects
    # Harmony
    harmony_chords = [Chord(**c) for c in content_data["harmony"]["chords"]]
    harmony = ChordProgression(
        track=content_data["harmony"].get("track", harmony_track),
        section=content_data["harmony"].get("section", section.name),
        chords=harmony_chords
    )

    # Melody (may be None for intro/outro)
    melody = None
    if "melody" in content_data and content_data["melody"] is not None:
        melody_notes = [Note(**n) for n in content_data["melody"]["notes"]]
        melody = MelodyPhrase(
            track=content_data["melody"].get("track", melody_track),
            section=content_data["melody"].get("section", section.name),
            notes=melody_notes,
            motif_id=content_data["melody"].get("motif_id")
        )

    # Bass
    bass_notes = [Note(**n) for n in content_data["bass"]["notes"]]
    bass = BassLine(
        track=content_data["bass"].get("track", bass_track),
        section=content_data["bass"].get("section", section.name),
        notes=bass_notes
    )

    print(f"[MUSICAL_CONTENT] Generated:")
    print(f"  - Harmony: {len(harmony_chords)} chords")
    if melody:
        print(f"  - Melody: {len(melody.notes)} notes")
    print(f"  - Bass: {len(bass_notes)} notes")

    return {
        "harmony": harmony,
        "melody": melody,
        "bass": bass
    }

