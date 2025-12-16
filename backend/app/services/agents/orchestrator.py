"""Orchestrator - Coordinates multi-agent composition pipeline."""

from app.services.mir.schema import StyleGuide, Section, ChordProgression
from app.services.agents.style_agent import invoke_style_agent
from app.services.agents.arranger_agent import invoke_arranger_agent
from app.services.agents.harmony_agent import invoke_harmony_agent
from app.services.mir.compiler import compile_progression_to_tool_calls
from typing import List, Dict


async def orchestrate_composition(
    user_prompt: str,
    target_bars: int = 32
) -> Dict:
    """
    Orchestrate full composition through subagents.

    This is the Phase 2 orchestrator that creates:
    1. Style guide (via Style Agent)
    2. Song structure (via Arranger Agent)
    3. Harmony for each section (via Harmony Agent)

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

    # Step 3: Harmony Agent (for each section)
    print(f"[ORCHESTRATOR] Step 3: Generating harmony for each section...")
    all_tool_calls = []
    track_id = 1  # Will be set dynamically when we create the track

    # First, create the piano track
    all_tool_calls.append({
        "name": "createTrack",
        "args": {"instrumentName": "piano", "trackName": "Piano"}
    })

    # Generate harmony for each section
    for i, section in enumerate(sections):
        print(f"[ORCHESTRATOR] Generating harmony for section {i+1}/{len(sections)}: {section.name}")
        progression = await invoke_harmony_agent(style_guide, section, "piano")
        tool_calls = compile_progression_to_tool_calls(progression, track_id)
        all_tool_calls.extend(tool_calls)

    print(f"[ORCHESTRATOR] Orchestration complete! Generated {len(all_tool_calls)} tool calls")

    return {
        "style_guide": style_guide,
        "sections": sections,
        "tool_calls": all_tool_calls
    }
