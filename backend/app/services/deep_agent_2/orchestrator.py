"""Orchestrator - Coordinates multi-agent composition pipeline (Deep Agent 2.0)."""

from app.services.mir.schema import StyleGuide, Section, ChordProgression, MelodyPhrase, DrumPattern, BassLine
from app.services.agents.style_agent import invoke_style_agent
from app.services.agents.arranger_agent import invoke_arranger_agent
from app.services.deep_agent_2.musical_content_agent import invoke_musical_content_agent
# Removed: melody_agent, rhythm_agent, bass_agent - now handled by musical_content_agent
from app.services.deep_agent_2.orchestration_agent import invoke_orchestration_agent
from app.services.agents.critic_agent import invoke_critic_agent
from app.services.mir.compiler import (
    compile_progression_to_tool_calls,
    compile_melody_to_notes,
    compile_drums_to_notes,
    compile_bass_to_notes
)
from typing import List, Dict, Optional


async def orchestrate_composition(
    user_prompt: str,
    target_bars: int = 32
) -> Dict:
    """
    Orchestrate full composition through subagents.

    Creates:
    1. Style guide (via Style Agent)
    2. Song structure (via Arranger Agent)
    3. Harmony for each section (via Harmony Agent)
    4. Melody for each section (via Melody Agent)
    5. Rhythm for each section (via Rhythm Agent)

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
    print(f"[DEEP_AGENT_2.0] Step 1: Invoking Style Agent...")
    style_guide = await invoke_style_agent(user_prompt)
    print(f"[DEEP_AGENT_2.0] Style: {style_guide.genre} {style_guide.subgenre}")

    # Step 2: Arranger Agent - create song structure
    print(f"[DEEP_AGENT_2.0] Step 2: Invoking Arranger Agent...")
    sections = await invoke_arranger_agent(style_guide, target_bars)
    print(f"[DEEP_AGENT_2.0] Created {len(sections)} sections")

    # Step 3: Generate content for each section
    print(f"[DEEP_AGENT_2.0] Step 3: Generating content for each section...")
    all_content = {}  # section_name → {harmony, melody, bass, rhythm}
    
    for i, section in enumerate(sections):
        print(f"[DEEP_AGENT_2.0] Section {i+1}/{len(sections)}: {section.name}")

        # Generate harmony, melody, and bass together
        # Note: Using generic placeholders - orchestration agent will assign actual instruments
        print(f"[DEEP_AGENT_2.0]   - Generating musical content (harmony, melody, bass)...")
        content = await invoke_musical_content_agent(style_guide, section, harmony_track="harmony_track", melody_track="melody_track", bass_track="bass_track")
        
        # Store content (melody may be None for intro/outro)
        all_content[section.name] = {
            "harmony": content["harmony"],
            "melody": content["melody"] if section.name not in ["intro", "outro"] else None,
            "bass": content["bass"],
            "rhythm": None  # Rhythm generation not yet implemented
        }

    # Step 4: Orchestration - assign instruments to content
    print(f"\n[DEEP_AGENT_2.0] Step 4: Invoking Orchestration Agent...")
    orchestration = await invoke_orchestration_agent(style_guide, sections, all_content)

    # Step 5: Compile to MIDI tool calls
    print(f"[DEEP_AGENT_2.0] Step 5: Compiling to MIDI tool calls...")
    all_tool_calls = []
    track_map = {}  # instrument_name → track_id

    # Create tracks based on orchestration plan
    for assignment in orchestration.assignments:
        instrument = assignment.instrument
        if instrument not in track_map:
            track_id = len(track_map) + 1
            track_map[instrument] = track_id
            # Capitalize instrument name for display
            track_name = instrument.replace("_", " ").title()
            all_tool_calls.append({
                "name": "createTrack",
                "args": {"instrumentName": instrument, "trackName": track_name}
            })
            print(f"[DEEP_AGENT_2.0]   Created track {track_id}: {track_name}")

    # Compile content with orchestration
    for section_name, content in all_content.items():
        print(f"[DEEP_AGENT_2.0]   Compiling section: {section_name}")

        # Find which instruments play each part in this section
        for assignment in orchestration.assignments:
            if section_name not in assignment.active_sections:
                continue  # This instrument doesn't play in this section

            track_id = track_map[assignment.instrument]
            source = assignment.source_content

            # Get the MIR content
            mir_content = content.get(source)
            if mir_content is None:
                continue

            # Compile based on content type
            if source == "harmony":
                tool_calls = compile_progression_to_tool_calls(mir_content, track_id)
                all_tool_calls.extend(tool_calls)
            elif source == "melody":
                notes = compile_melody_to_notes(mir_content)
                all_tool_calls.append({
                    "name": "addNotes",
                    "args": {"trackId": track_id, "notes": notes}
                })
            elif source == "bass":
                notes = compile_bass_to_notes(mir_content)
                all_tool_calls.append({
                    "name": "addNotes",
                    "args": {"trackId": track_id, "notes": notes}
                })
            elif source == "rhythm":
                if mir_content is not None:
                    notes = compile_drums_to_notes(mir_content)
                    all_tool_calls.append({
                        "name": "addNotes",
                        "args": {"trackId": track_id, "notes": notes}
                    })

    print(f"[DEEP_AGENT_2.0] Orchestration complete! Generated {len(all_tool_calls)} tool calls")

    return {
        "style_guide": style_guide,
        "sections": sections,
        "tool_calls": all_tool_calls
    }

