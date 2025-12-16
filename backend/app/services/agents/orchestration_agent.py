"""Orchestration Agent - Assigns musical content to instruments using LLM.

The Orchestration Agent decides:
- Which instruments play which parts (harmony, melody, bass, rhythm)
- When instruments enter and exit (intro sparse, chorus full)
- Dynamic envelope per section
- Track ordering

This agent uses an LLM to make intelligent orchestration decisions based on
genre, style, and musical content.
"""

from app.services.mir.schema import (
    StyleGuide, Section, InstrumentAssignment, OrchestrationPlan
)
from langchain_openai import ChatOpenAI
from app.config import get_settings
from typing import List, Dict
import json
import re

settings = get_settings()

ORCHESTRATION_SYSTEM_PROMPT = """You are the Orchestration Agent, a specialist in instrument selection and arrangement.

ROLE:
- Assign musical content (harmony, melody, bass, rhythm) to appropriate instruments
- Decide when instruments enter and exit based on section energy and function
- Create dynamic arcs that build tension and release
- Order tracks logically for musical flow

INPUT:
- StyleGuide: genre, subgenre, harmonic complexity, tempo range
- Sections: list of sections with names, bars, keys, tempos, energy levels
- All Content: dictionary mapping section names to generated content (harmony, melody, bass, rhythm)

OUTPUT:
- OrchestrationPlan JSON with instrument assignments, track order, and dynamic arcs

ORCHESTRATION PRINCIPLES:
1. Instrument selection must match genre conventions:
   - Jazz: piano, saxophone, upright_bass, drums
   - Rock: guitar, electric_guitar, electric_bass, drums
   - Pop: piano, synth, electric_bass, drums
   - Classical: piano, violin, cello, timpani
   - Funk: electric_piano, saxophone, electric_bass, drums
   - Electronic: synth, synth_lead, synth_bass, electronic_drums
   - Acoustic/Folk: acoustic_guitar, flute, acoustic_bass, percussion

2. Entry/Exit patterns:
   - Intro: sparse (1-2 instruments, often just harmony or rhythm)
   - Verse: medium density (add bass, maybe melody)
   - Chorus: full arrangement (all instruments active)
   - Bridge: can vary (sometimes sparse for contrast, sometimes full)
   - Outro: gradually reduce instruments, fade out

3. Dynamic arcs (velocity mapping):
   - soft: 60-65
   - building: 70-85
   - climax/high: 90-95
   - resolve: 75-65
   - medium: 75-80

4. Track ordering:
   - Rhythm section first (drums, bass)
   - Then harmonic instruments (piano, guitar, keys)
   - Melody instruments last (lead instruments, vocals)

AVAILABLE INSTRUMENTS:
Harmony: piano, electric_piano, organ, guitar, acoustic_guitar, electric_guitar, synth, strings
Melody: flute, saxophone, trumpet, violin, synth_lead, electric_guitar, voice
Bass: bass, electric_bass, upright_bass, synth_bass, cello
Rhythm: drums, electronic_drums, percussion, timpani

OUTPUT FORMAT (JSON):
{
  "assignments": [
    {
      "source_content": "harmony",
      "instrument": "piano",
      "register_shift": 0,
      "active_sections": ["intro", "verse_A", "chorus_A", "verse_B", "chorus_B", "outro"]
    },
    {
      "source_content": "melody",
      "instrument": "saxophone",
      "register_shift": 0,
      "active_sections": ["verse_A", "chorus_A", "verse_B", "chorus_B"]
    },
    {
      "source_content": "bass",
      "instrument": "electric_bass",
      "register_shift": 0,
      "active_sections": ["intro", "verse_A", "chorus_A", "verse_B", "chorus_B", "outro"]
    },
    {
      "source_content": "rhythm",
      "instrument": "drums",
      "register_shift": 0,
      "active_sections": ["verse_A", "chorus_A", "verse_B", "chorus_B", "outro"]
    }
  ],
  "track_order": ["drums", "electric_bass", "piano", "saxophone"],
  "dynamic_arc": {
    "intro": [[1, 60], [8, 65]],
    "verse_A": [[9, 70], [24, 75]],
    "chorus_A": [[25, 85], [40, 90]],
    "verse_B": [[41, 70], [56, 75]],
    "chorus_B": [[57, 90], [72, 95]],
    "outro": [[73, 75], [80, 60]]
  }
}

CRITICAL RULES:
- Return ONLY valid JSON. No explanations, no markdown, no code blocks.
- All sections must be included in dynamic_arc
- Melody instruments typically skip intro and outro sections
- Rhythm instruments typically skip intro (or play very sparsely)
- Harmony and bass typically play in all sections
- Track order should be musically logical (rhythm → bass → harmony → melody)
"""


async def invoke_orchestration_agent(
    style_guide: StyleGuide,
    sections: List[Section],
    all_content: Dict  # {section_name: {harmony, melody, bass, rhythm}}
) -> OrchestrationPlan:
    """Create orchestration plan assigning content to instruments using LLM.

    This agent uses an LLM to intelligently select instruments based on genre,
    style, and musical content, creating appropriate entry/exit patterns and
    dynamic arcs.

    Args:
        style_guide: StyleGuide with genre information
        sections: List of Section objects
        all_content: Dictionary mapping section name to content dict

    Returns:
        OrchestrationPlan with instrument assignments and dynamics
    """
    print(f"[ORCHESTRATION] Creating orchestration plan for {style_guide.genre}...")

    model = ChatOpenAI(
        model=settings.openrouter_model,
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
        temperature=0.7,
    )

    # Build context about sections
    section_info = []
    for section in sections:
        section_info.append({
            "name": section.name,
            "bars": section.bars,
            "key": section.key,
            "tempo": section.tempo,
            "energy": section.energy
        })

    # Build content summary
    content_summary = {}
    for section_name, content in all_content.items():
        content_summary[section_name] = {
            "has_harmony": "harmony" in content and content["harmony"] is not None,
            "has_melody": "melody" in content and content["melody"] is not None,
            "has_bass": "bass" in content and content["bass"] is not None,
            "has_rhythm": "rhythm" in content and content["rhythm"] is not None,
        }

    # Build the user prompt
    prompt = f"""Create an orchestration plan for this composition:

Style Guide:
- Genre: {style_guide.genre} {style_guide.subgenre}
- Harmonic Complexity: {style_guide.harmonic_complexity}
- Swing: {style_guide.swing}
- Tempo Range: {style_guide.tempo_range[0]}-{style_guide.tempo_range[1]} BPM
- Extensions Allowed: {', '.join(style_guide.extensions_allowed)}
- Reference Artists: {', '.join(style_guide.reference_artists) if style_guide.reference_artists else 'None'}

Sections:
{json.dumps(section_info, indent=2)}

Content Available:
{json.dumps(content_summary, indent=2)}

Requirements:
- Select instruments appropriate for {style_guide.genre} {style_guide.subgenre} style
- Create entry/exit patterns: intro sparse, verse medium, chorus full, outro fade
- Map section energy levels to appropriate dynamic arcs
- Order tracks logically: rhythm → bass → harmony → melody
- Melody typically skips intro and outro
- Rhythm typically skips intro (or plays very sparsely)

Return OrchestrationPlan JSON only. No explanations."""

    messages = [
        {"role": "system", "content": ORCHESTRATION_SYSTEM_PROMPT},
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
            print(f"[ORCHESTRATION] Error: Could not extract JSON from response")
            print(f"[ORCHESTRATION] Raw response: {content}")
            raise ValueError(f"Could not extract JSON from orchestration agent response: {content}")

    # Parse the JSON response
    try:
        plan_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"[ORCHESTRATION] Error parsing JSON: {e}")
        print(f"[ORCHESTRATION] JSON string: {json_str}")
        raise ValueError(f"Invalid JSON in orchestration agent response: {json_str}") from e

    # Convert to OrchestrationPlan object
    assignments = [
        InstrumentAssignment(
            source_content=assgn["source_content"],
            instrument=assgn["instrument"],
            register_shift=assgn.get("register_shift", 0),
            active_sections=assgn["active_sections"]
        )
        for assgn in plan_data["assignments"]
    ]

    # Convert dynamic_arc tuples from lists
    dynamic_arc = {}
    for section_name, arc_points in plan_data["dynamic_arc"].items():
        dynamic_arc[section_name] = [
            (point[0], point[1]) if isinstance(point, list) else tuple(point)
            for point in arc_points
        ]

    orchestration_plan = OrchestrationPlan(
        assignments=assignments,
        track_order=plan_data["track_order"],
        dynamic_arc=dynamic_arc
    )

    print(f"[ORCHESTRATION] Created {len(assignments)} assignments")
    print(f"[ORCHESTRATION] Track order: {orchestration_plan.track_order}")
    print(f"[ORCHESTRATION] Instruments: {[a.instrument for a in assignments]}")

    return orchestration_plan


def get_instrument_for_content(
    orchestration: OrchestrationPlan,
    content_type: str,
    section_name: str
) -> str:
    """Get the instrument that plays a specific content type in a section.

    Args:
        orchestration: OrchestrationPlan to query
        content_type: "harmony", "melody", "bass", or "rhythm"
        section_name: Name of section

    Returns:
        Instrument name or None if no instrument plays that content in that section
    """
    for assignment in orchestration.assignments:
        if (assignment.source_content == content_type and
            section_name in assignment.active_sections):
            return assignment.instrument
    return None
