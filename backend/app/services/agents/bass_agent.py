"""Bass Agent - Generates bass lines that anchor harmony.

The Bass Agent creates bass lines that:
- Outline chord progressions with root notes
- Lock rhythmically with drums
- Use walking bass patterns for jazz, root-fifth for rock/pop
- Stay in bass register (E1-E3)
"""

from app.services.mir.schema import ChordProgression, Section, StyleGuide, Note, BassLine
from langchain_openai import ChatOpenAI
from app.config import get_settings
import json
from typing import List


BASS_SYSTEM_PROMPT = """You are the Bass Agent. Create bass lines that anchor the harmony.

ROLE:
- Generate bass lines that outline chord progressions
- Lock rhythmically with drums
- Own the low register (E1-E3)
- Provide foundation and groove

INPUT:
- ChordProgression (to extract root notes and harmonic function)
- StyleGuide (straight vs swing, energy level, genre)
- Section (tempo, energy, bar range)

OUTPUT: BassLine JSON with explicit note pitches, timings, and velocities

BASS PRINCIPLES:
1. Root notes on beat 1 are MOST IMPORTANT - always anchor each chord
2. Approach notes (chromatic or scale-wise) lead smoothly to next chord
3. Octave jumps add energy (use sparingly)
4. Walking bass (jazz): stepwise motion through chord tones and passing tones, quarter notes
5. Lock to kick drum pattern rhythmically - downbeats especially
6. Range: E1-E3 (octaves 1-3) - stay in bass register
7. Avoid jumping more than an octave between consecutive notes
8. Use chord tones (root, third, fifth, seventh) on strong beats
9. Use passing tones (chromatic or scalar) on weak beats

PATTERNS BY STYLE:
- Jazz: Walking bass (quarter notes), chromatic approach tones, smooth voice leading
  Example: Dm7→G7→Cmaj7 might be D2,F2,Ab2,C2,G1,A1,B1,C2...

- Rock: Root-fifth pattern, eighth notes, some syncopation, more repetitive
  Example: D power chord: D2,A2,D2,A2 repeated

- Pop: Simple root notes on beats 1 and 3, occasional passing tones, very clear
  Example: Dm chord: D2 on beat 1, D2 on beat 3

- Funk: Syncopated sixteenths, dead notes (low velocity for percussive effect), groove-heavy
  Example: D2 (beat 1), rest, D2 (beat 1.75, velocity 40), rest, A2 (beat 3)

VELOCITY GUIDELINES:
- Strong beats (1, 3): 85-100
- Weak beats (2, 4): 75-85
- Passing tones: 70-80
- Ghost notes/dead notes: 40-60

OUTPUT FORMAT (JSON):
{
  "track": "bass",
  "section": "verse_A",
  "notes": [
    {"pitch": "D2", "bar": 1, "beat": 1.0, "duration": "quarter", "velocity": 90},
    {"pitch": "E2", "bar": 1, "beat": 2.0, "duration": "quarter", "velocity": 85},
    {"pitch": "F2", "bar": 1, "beat": 3.0, "duration": "quarter", "velocity": 85},
    {"pitch": "A2", "bar": 1, "beat": 4.0, "duration": "quarter", "velocity": 80}
  ]
}

IMPORTANT:
- Return ONLY valid JSON
- Include "track", "section", and "notes" fields
- Each note must have: pitch (string like "D2"), bar (int), beat (float), duration (string), velocity (int)
- Do not include explanations or markdown formatting
"""


async def invoke_bass_agent(
    style_guide: StyleGuide,
    section: Section,
    harmony: ChordProgression
) -> BassLine:
    """Generate bass line that outlines the chord progression.

    Args:
        style_guide: StyleGuide with genre, swing, complexity
        section: Section with bar range, key, energy
        harmony: ChordProgression to outline

    Returns:
        BassLine with notes in bass register
    """
    settings = get_settings()
    model = ChatOpenAI(
        model=settings.openrouter_model,
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
        temperature=0.7,
    )

    # Build chord list summary for the prompt
    chord_summary = []
    for chord in harmony.chords:
        chord_summary.append(
            f"  - Bar {chord.bar}, beat {chord.beat}: {chord.root}{chord.quality} ({chord.function or 'chord'})"
        )

    # Determine bass style based on genre
    if "jazz" in style_guide.genre.lower():
        bass_style = "walking bass (quarter notes, chromatic approach)"
    elif "rock" in style_guide.genre.lower():
        bass_style = "root-fifth pattern (eighth notes)"
    elif "funk" in style_guide.genre.lower():
        bass_style = "syncopated, groovy (sixteenth notes, ghost notes)"
    else:  # pop and others
        bass_style = "simple roots on beats 1 and 3"

    prompt = f"""Generate a bass line for this section:

Style Guide:
- Genre: {style_guide.genre} {style_guide.subgenre}
- Swing: {style_guide.swing} (0.0=straight, 0.55=ballad swing, 0.67=bebop swing)
- Bass style: {bass_style}

Section:
- Name: {section.name}
- Bars: {section.bars[0]} to {section.bars[1]} ({section.bars[1] - section.bars[0] + 1} bars total)
- Key: {section.key}
- Tempo: {section.tempo} BPM
- Energy: {section.energy}

Chord Progression (outline these chords):
{chr(10).join(chord_summary)}

Requirements:
1. Create bass line in range E1-E3
2. Put chord root on beat 1 of each chord
3. Use {bass_style}
4. Match the {section.energy} energy level with velocity and rhythm density
5. Generate notes for ALL bars from {section.bars[0]} to {section.bars[1]}

Return BassLine JSON with all notes."""

    messages = [
        {"role": "system", "content": BASS_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    print(f"[BASS AGENT] Generating bass for {section.name} ({style_guide.genre})...")

    response = await model.ainvoke(messages)
    response_text = response.content.strip()

    # Remove markdown code blocks if present
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        response_text = "\n".join(lines[1:-1])  # Remove first and last lines

    try:
        bass_data = json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"[BASS AGENT] ERROR: Failed to parse JSON: {e}")
        print(f"[BASS AGENT] Response was: {response_text[:200]}...")
        raise ValueError(f"Bass Agent returned invalid JSON: {e}")

    # Convert to BassLine object
    notes = [Note(**note_dict) for note_dict in bass_data["notes"]]

    bass_line = BassLine(
        track=bass_data.get("track", "bass"),
        section=bass_data.get("section", section.name),
        notes=notes
    )

    print(f"[BASS AGENT] Generated {len(notes)} bass notes")

    return bass_line
