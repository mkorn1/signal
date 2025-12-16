"""Arranger Agent - Creates song structure and energy arc."""

from langchain_openai import ChatOpenAI
from app.services.mir.schema import StyleGuide, Section
from app.config import get_settings
from typing import List
import json
import re

settings = get_settings()

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
    "energy": "soft"
  },
  {
    "name": "verse_A",
    "bars": [9, 24],
    "key": "Dm",
    "tempo": 72,
    "energy": "building"
  }
]

Return ONLY valid JSON array. No markdown formatting or explanations."""


async def invoke_arranger_agent(
    style_guide: StyleGuide,
    target_length_bars: int = 80
) -> List[Section]:
    """Invoke Arranger Agent to create song structure.

    Args:
        style_guide: StyleGuide defining musical style
        target_length_bars: Desired total length in bars

    Returns:
        List of Section objects defining song structure
    """
    model = ChatOpenAI(
        model=settings.openrouter_model,
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
        temperature=0.7,
    )

    # Build prompt
    prompt = f"""Create a song structure for this style:

Style Guide:
- Genre: {style_guide.genre} {style_guide.subgenre}
- Tempo range: {style_guide.tempo_range[0]}-{style_guide.tempo_range[1]} BPM
- Target length: approximately {target_length_bars} bars

Requirements:
- Use appropriate form for {style_guide.genre} (see templates)
- Create clear energy arc (not flat - should build and release)
- Contrast between sections
- Total bars should be close to {target_length_bars}

Return Section array JSON only."""

    messages = [
        {"role": "system", "content": ARRANGER_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    response = await model.ainvoke(messages)
    content = response.content

    # Extract JSON from response
    json_match = re.search(r'```(?:json)?\s*(\[.*\])\s*```', content, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find JSON array directly
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            raise ValueError(f"Could not extract JSON array from response: {content}")

    try:
        sections_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in response: {json_str}") from e

    # Convert to Section objects
    sections = []
    for section_data in sections_data:
        # Convert bars list to tuple
        if isinstance(section_data.get("bars"), list):
            section_data["bars"] = tuple(section_data["bars"])

        sections.append(Section(**section_data))

    return sections
