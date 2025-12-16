"""Melody Agent - Creates melodic lines that fit harmony and style."""

from app.services.mir.schema import StyleGuide, Section, ChordProgression, Note, MelodyPhrase
from langchain_openai import ChatOpenAI
from app.config import get_settings
import json

settings = get_settings()

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
    {"pitch": "E4", "bar": 1, "beat": 2.5, "duration": "eighth", "velocity": 70}
  ],
  "motif_id": "motif_A"
}

CRITICAL RULES:
- Return ONLY valid JSON. No explanations, no markdown, no code blocks.
- All notes must be within the range C4-G5
- Use chord tones on strong beats (beat 1.0, 3.0)
- Create phrases with natural breathing points (rests)
- Vary rhythm to avoid monotony
"""


async def invoke_melody_agent(
    style_guide: StyleGuide,
    section: Section,
    harmony: ChordProgression,
    track_name: str = "flute"
) -> MelodyPhrase:
    """
    Invoke the melody agent to generate a melodic phrase.

    Args:
        style_guide: Style guide for the composition
        section: Section to generate melody for
        harmony: Chord progression to fit melody to
        track_name: Instrument name for the melody track

    Returns:
        MelodyPhrase MIR object
    """
    model = ChatOpenAI(
        model=settings.openrouter_model,
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
        temperature=0.8,  # Higher temperature for more creative melodies
    )

    # Build the user prompt with context
    chord_info = []
    for chord in harmony.chords:
        chord_info.append(f"Bar {chord.bar}: {chord.root}{chord.quality} ({chord.function or 'N/A'})")

    prompt = f"""Generate a melody for this section:

Style Guide:
- Genre: {style_guide.genre} {style_guide.subgenre}
- Harmonic Complexity: {style_guide.harmonic_complexity}
- Swing: {style_guide.swing}
- Extensions Allowed: {', '.join(style_guide.extensions_allowed)}

Section:
- Name: {section.name}
- Bars: {section.bars[0]}-{section.bars[1]}
- Key: {section.key}
- Tempo: {section.tempo}
- Energy: {section.energy}

Chord Progression:
{chr(10).join(chord_info)}

Track: {track_name}

Requirements:
- Generate a melody that fits the {style_guide.genre} style
- Use the chord tones from the progression above
- Create a memorable melodic contour
- Keep melody in range C4-G5
- Match the "{section.energy}" energy level
- Use appropriate phrasing with breathing room

Return MelodyPhrase JSON only. No explanations."""

    messages = [
        {"role": "system", "content": MELODY_SYSTEM_PROMPT},
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

    # Parse the JSON response
    try:
        phrase_data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"[MELODY_AGENT] Error parsing JSON: {e}")
        print(f"[MELODY_AGENT] Raw response: {content}")
        raise ValueError(f"Failed to parse melody agent response as JSON: {e}")

    # Convert to MelodyPhrase object
    notes = [Note(**n) for n in phrase_data["notes"]]
    phrase = MelodyPhrase(
        track=phrase_data.get("track", track_name),
        section=phrase_data["section"],
        notes=notes,
        motif_id=phrase_data.get("motif_id")
    )

    print(f"[MELODY_AGENT] Generated melody with {len(notes)} notes for section {section.name}")

    return phrase
