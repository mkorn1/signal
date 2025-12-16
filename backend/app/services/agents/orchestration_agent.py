"""Orchestration Agent - Assigns musical content to instruments.

The Orchestration Agent decides:
- Which instruments play which parts (harmony, melody, bass, rhythm)
- When instruments enter and exit (intro sparse, chorus full)
- Dynamic envelope per section
- Track ordering

This is a rule-based MVP for Phase 5. Future versions may use LLM for more creative choices.
"""

from app.services.mir.schema import (
    StyleGuide, Section, InstrumentAssignment, OrchestrationPlan
)
from typing import List, Dict


# Instrument choices by genre
GENRE_INSTRUMENTS = {
    "jazz": {
        "harmony": "piano",
        "melody": "saxophone",
        "bass": "upright_bass",
        "rhythm": "drums"
    },
    "rock": {
        "harmony": "guitar",
        "melody": "electric_guitar",
        "bass": "electric_bass",
        "rhythm": "drums"
    },
    "pop": {
        "harmony": "piano",
        "melody": "synth",
        "bass": "electric_bass",
        "rhythm": "drums"
    },
    "classical": {
        "harmony": "piano",
        "melody": "violin",
        "bass": "cello",
        "rhythm": "timpani"
    },
    "funk": {
        "harmony": "electric_piano",
        "melody": "saxophone",
        "bass": "electric_bass",
        "rhythm": "drums"
    },
    "default": {
        "harmony": "piano",
        "melody": "flute",
        "bass": "bass",
        "rhythm": "drums"
    }
}


async def invoke_orchestration_agent(
    style_guide: StyleGuide,
    sections: List[Section],
    all_content: Dict  # {section_name: {harmony, melody, bass, rhythm}}
) -> OrchestrationPlan:
    """Create orchestration plan assigning content to instruments.

    This rule-based MVP assigns instruments based on genre and creates
    entry/exit patterns for dynamic arc.

    Args:
        style_guide: StyleGuide with genre information
        sections: List of Section objects
        all_content: Dictionary mapping section name to content dict

    Returns:
        OrchestrationPlan with instrument assignments and dynamics
    """
    print(f"[ORCHESTRATION] Creating orchestration plan for {style_guide.genre}...")

    # Select instruments based on genre
    genre_lower = style_guide.genre.lower()
    instruments = GENRE_INSTRUMENTS.get(
        genre_lower,
        GENRE_INSTRUMENTS["default"]
    )

    print(f"[ORCHESTRATION] Instrument assignments: {instruments}")

    # Create assignments
    assignments = []

    # Determine active sections for each part based on section type
    section_names = [s.name for s in sections]

    # Harmony: play in all sections
    assignments.append(InstrumentAssignment(
        source_content="harmony",
        instrument=instruments["harmony"],
        register_shift=0,
        active_sections=section_names.copy()
    ))

    # Melody: skip intro and outro
    melody_sections = [
        name for name in section_names
        if name not in ["intro", "outro"]
    ]
    assignments.append(InstrumentAssignment(
        source_content="melody",
        instrument=instruments["melody"],
        register_shift=0,
        active_sections=melody_sections
    ))

    # Bass: play in all sections
    assignments.append(InstrumentAssignment(
        source_content="bass",
        instrument=instruments["bass"],
        register_shift=0,
        active_sections=section_names.copy()
    ))

    # Rhythm: skip intro, play in everything else
    rhythm_sections = [
        name for name in section_names
        if name != "intro"
    ]
    assignments.append(InstrumentAssignment(
        source_content="rhythm",
        instrument=instruments["rhythm"],
        register_shift=0,
        active_sections=rhythm_sections
    ))

    # Track order: rhythm section first, then melody
    track_order = [
        instruments["rhythm"],
        instruments["bass"],
        instruments["harmony"],
        instruments["melody"]
    ]

    # Create dynamic arc based on section energy
    dynamic_arc = {}
    for section in sections:
        start_bar = section.bars[0]
        end_bar = section.bars[1]

        # Map energy to velocity
        if section.energy == "soft":
            start_vel = 60
            end_vel = 65
        elif section.energy == "building":
            start_vel = 70
            end_vel = 85
        elif section.energy == "climax" or section.energy == "high":
            start_vel = 90
            end_vel = 95
        elif section.energy == "resolve":
            start_vel = 75
            end_vel = 65
        else:  # medium
            start_vel = 75
            end_vel = 80

        dynamic_arc[section.name] = [
            (start_bar, start_vel),
            (end_bar, end_vel)
        ]

    print(f"[ORCHESTRATION] Created {len(assignments)} assignments")
    print(f"[ORCHESTRATION] Track order: {track_order}")

    orchestration_plan = OrchestrationPlan(
        assignments=assignments,
        track_order=track_order,
        dynamic_arc=dynamic_arc
    )

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
