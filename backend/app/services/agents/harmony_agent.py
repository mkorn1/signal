"""Harmony Agent - Generates chord progressions with proper voice leading."""

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from app.services.mir.schema import StyleGuide, Section, ChordProgression, Chord
from app.config import get_settings
import json
import re

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
    }
  ]
}

Return ONLY valid JSON. No explanations or markdown formatting."""


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

    Args:
        style_guide: StyleGuide defining musical style constraints
        section: Section defining bars, key, energy
        track_name: Target instrument (default: "piano")

    Returns:
        ChordProgression MIR object
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

    # Try to extract JSON from markdown code blocks or plain text
    json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', last_message, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find JSON directly
        json_match = re.search(r'\{.*\}', last_message, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            raise ValueError(f"Could not extract JSON from response: {last_message}")

    # Parse the JSON response
    try:
        progression_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in response: {json_str}") from e

    # Convert to ChordProgression object
    chords = [Chord(**c) for c in progression_data["chords"]]
    progression = ChordProgression(
        track=progression_data["track"],
        section=progression_data["section"],
        chords=chords
    )

    return progression
