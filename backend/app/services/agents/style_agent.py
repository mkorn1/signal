"""Style Agent - Establishes musical DNA for compositions."""

from langchain_openai import ChatOpenAI
from app.services.mir.schema import StyleGuide
from app.config import get_settings
import json
import re

settings = get_settings()

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

Return ONLY valid JSON. No markdown formatting or explanations."""


async def invoke_style_agent(user_style_description: str) -> StyleGuide:
    """Invoke Style Agent to create StyleGuide from user description.

    Args:
        user_style_description: User's description of desired style

    Returns:
        StyleGuide object with all stylistic constraints
    """
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

    # Extract JSON from response
    content = response.content

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
            raise ValueError(f"Could not extract JSON from response: {content}")

    try:
        style_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in response: {json_str}") from e

    # Convert tempo_range to tuple if it's a list
    if isinstance(style_data.get("tempo_range"), list):
        style_data["tempo_range"] = tuple(style_data["tempo_range"])

    return StyleGuide(**style_data)
