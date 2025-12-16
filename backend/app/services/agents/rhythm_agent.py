"""Rhythm Agent - Creates drum patterns and grooves."""

from app.services.mir.schema import StyleGuide, Section, DrumPattern, DrumHit
from langchain_openai import ChatOpenAI
from app.config import get_settings
import json

settings = get_settings()

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

DRUM MAPPING (use these exact names):
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
    {"instrument": "hihat_closed", "bar": 1, "beat": 1.5, "velocity": 60}
  ]
}

CRITICAL RULES:
- Return ONLY valid JSON. No explanations, no markdown, no code blocks.
- Use only the drum instruments listed above (kick, snare, hihat_closed, hihat_open, crash, ride)
- Match the swing amount from the StyleGuide
- Kick on beat 1.0 is standard
- Snare on beats 2.0 and 4.0 for backbeat
- Hi-hat provides subdivision (typically on every beat or eighth note)
- Adjust velocity to match energy level (soft=60-75, medium=75-90, high=90-110)
"""


async def invoke_rhythm_agent(
    style_guide: StyleGuide,
    section: Section,
    track_name: str = "drums"
) -> DrumPattern:
    """
    Invoke the rhythm agent to generate a drum pattern.

    Args:
        style_guide: Style guide for the composition
        section: Section to generate rhythm for
        track_name: Track name for drums (default "drums")

    Returns:
        DrumPattern MIR object
    """
    model = ChatOpenAI(
        model=settings.openrouter_model,
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
        temperature=0.7,
    )

    # Map energy to velocity range
    energy_velocity_map = {
        "soft": (60, 80),
        "building": (75, 95),
        "medium": (80, 100),
        "climax": (90, 110),
        "high": (90, 110),
        "resolve": (70, 90)
    }
    velocity_range = energy_velocity_map.get(section.energy, (75, 95))

    prompt = f"""Generate a drum pattern for this section:

Style Guide:
- Genre: {style_guide.genre} {style_guide.subgenre}
- Swing: {style_guide.swing}

Section:
- Name: {section.name}
- Bars: {section.bars[0]}-{section.bars[1]}
- Key: {section.key}
- Tempo: {section.tempo}
- Energy: {section.energy}

Track: {track_name}

Requirements:
- Create a {style_guide.genre} style drum pattern
- Use swing amount: {style_guide.swing}
- Match the "{section.energy}" energy level
- Velocity range for this energy: {velocity_range[0]}-{velocity_range[1]}
- Generate a pattern for bars {section.bars[0]}-{section.bars[1]}
- Typical patterns:
  * Kick: beats 1, 3 (rock/pop) or 1, 2.5, 4 (funk) or all 4 beats (jazz walking)
  * Snare: beats 2, 4 (backbeat)
  * Hi-hat: eighth notes or quarter notes depending on tempo and style
  * Crash: beginning of section for emphasis
  * Ride: jazz styles instead of hi-hat

Return DrumPattern JSON only. No explanations."""

    messages = [
        {"role": "system", "content": RHYTHM_SYSTEM_PROMPT},
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
        pattern_data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"[RHYTHM_AGENT] Error parsing JSON: {e}")
        print(f"[RHYTHM_AGENT] Raw response: {content}")
        raise ValueError(f"Failed to parse rhythm agent response as JSON: {e}")

    # Convert to DrumPattern object
    hits = [DrumHit(**h) for h in pattern_data["hits"]]
    pattern = DrumPattern(
        track=pattern_data.get("track", track_name),
        section=pattern_data["section"],
        hits=hits,
        swing=pattern_data.get("swing", style_guide.swing),
        variation_every_n_bars=pattern_data.get("variation_every_n_bars", 4)
    )

    print(f"[RHYTHM_AGENT] Generated drum pattern with {len(hits)} hits for section {section.name}")

    return pattern
