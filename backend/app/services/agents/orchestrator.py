"""Orchestrator - Coordinates multi-agent composition pipeline."""

from app.services.mir.schema import StyleGuide, Section, ChordProgression
from app.services.agents.style_agent import invoke_style_agent
from app.services.agents.arranger_agent import invoke_arranger_agent
from app.services.agents.harmony_agent import invoke_harmony_agent
from app.services.agents.melody_agent import invoke_melody_agent
from app.services.agents.rhythm_agent import invoke_rhythm_agent
from app.services.mir.compiler import (
    compile_progression_to_tool_calls,
    compile_melody_to_notes,
    compile_drums_to_notes
)
from typing import List, Dict


async def orchestrate_composition(
    user_prompt: str,
    target_bars: int = 32
) -> Dict:
    """
    Orchestrate full composition through subagents.

    This is the Phase 3 orchestrator that creates:
    1. Style guide (via Style Agent)
    2. Song structure (via Arranger Agent)
    3. Harmony for each section (via Harmony Agent)
    4. Melody for each section (via Melody Agent) - NEW in Phase 3
    5. Rhythm for each section (via Rhythm Agent) - NEW in Phase 3

    Returns compiled tool calls ready for frontend execution.

    Args:
        user_prompt: User's description of desired composition
        target_bars: Target length in bars (default 32)

    Returns:
        Dictionary with:
        - style_guide: StyleGuide object
        - sections: List of Section objects
        - tool_calls: List of tool call dictionaries for frontend
    """
    # Step 1: Style Agent - establish musical DNA
    print(f"[ORCHESTRATOR] Step 1: Invoking Style Agent...")
    style_guide = await invoke_style_agent(user_prompt)
    print(f"[ORCHESTRATOR] Style: {style_guide.genre} {style_guide.subgenre}")

    # Step 2: Arranger Agent - create song structure
    print(f"[ORCHESTRATOR] Step 2: Invoking Arranger Agent...")
    sections = await invoke_arranger_agent(style_guide, target_bars)
    print(f"[ORCHESTRATOR] Created {len(sections)} sections")

    # Track IDs
    piano_track_id = 1
    melody_track_id = 2
    drums_track_id = 3

    all_tool_calls = []

    # Create tracks
    print(f"[ORCHESTRATOR] Creating tracks...")
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

    # Step 3-5: Generate content for each section
    print(f"[ORCHESTRATOR] Generating content for each section...")
    for i, section in enumerate(sections):
        print(f"[ORCHESTRATOR] Section {i+1}/{len(sections)}: {section.name}")

        # Step 3: Harmony
        print(f"[ORCHESTRATOR]   - Generating harmony...")
        harmony = await invoke_harmony_agent(style_guide, section, "piano")
        harmony_calls = compile_progression_to_tool_calls(harmony, piano_track_id)
        all_tool_calls.extend(harmony_calls)

        # Step 4: Melody (only in verse/chorus, not intro/outro)
        if section.name not in ["intro", "outro"]:
            print(f"[ORCHESTRATOR]   - Generating melody...")
            melody = await invoke_melody_agent(style_guide, section, harmony, "flute")
            melody_notes = compile_melody_to_notes(melody)
            all_tool_calls.append({
                "name": "addNotes",
                "args": {"trackId": melody_track_id, "notes": melody_notes}
            })
        else:
            print(f"[ORCHESTRATOR]   - Skipping melody for {section.name}")

        # Step 5: Rhythm
        print(f"[ORCHESTRATOR]   - Generating rhythm...")
        rhythm = await invoke_rhythm_agent(style_guide, section)
        drum_notes = compile_drums_to_notes(rhythm)
        all_tool_calls.append({
            "name": "addNotes",
            "args": {"trackId": drums_track_id, "notes": drum_notes}
        })

    print(f"[ORCHESTRATOR] Orchestration complete! Generated {len(all_tool_calls)} tool calls")

    return {
        "style_guide": style_guide,
        "sections": sections,
        "tool_calls": all_tool_calls
    }
