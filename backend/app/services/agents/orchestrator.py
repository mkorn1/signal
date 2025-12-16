"""Orchestrator - Coordinates multi-agent composition pipeline."""

from app.services.mir.schema import StyleGuide, Section, ChordProgression, MelodyPhrase, DrumPattern, BassLine
from app.services.agents.style_agent import invoke_style_agent
from app.services.agents.arranger_agent import invoke_arranger_agent
from app.services.agents.musical_content_agent import invoke_musical_content_agent
# Removed: melody_agent, rhythm_agent, bass_agent - now handled by musical_content_agent
from app.services.agents.orchestration_agent import invoke_orchestration_agent
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
    print(f"[ORCHESTRATOR] Step 1: Invoking Style Agent...")
    style_guide = await invoke_style_agent(user_prompt)
    print(f"[ORCHESTRATOR] Style: {style_guide.genre} {style_guide.subgenre}")

    # Step 2: Arranger Agent - create song structure
    print(f"[ORCHESTRATOR] Step 2: Invoking Arranger Agent...")
    sections = await invoke_arranger_agent(style_guide, target_bars)
    print(f"[ORCHESTRATOR] Created {len(sections)} sections")

    # Step 3: Generate content for each section
    print(f"[ORCHESTRATOR] Step 3: Generating content for each section...")
    all_content = {}  # section_name → {harmony, melody, bass, rhythm}
    
    for i, section in enumerate(sections):
        print(f"[ORCHESTRATOR] Section {i+1}/{len(sections)}: {section.name}")

        # Generate harmony, melody, and bass together
        # Note: Using generic placeholders - orchestration agent will assign actual instruments
        print(f"[ORCHESTRATOR]   - Generating musical content (harmony, melody, bass)...")
        content = await invoke_musical_content_agent(style_guide, section, harmony_track="harmony_track", melody_track="melody_track", bass_track="bass_track")
        
        # Store content (melody may be None for intro/outro)
        all_content[section.name] = {
            "harmony": content["harmony"],
            "melody": content["melody"] if section.name not in ["intro", "outro"] else None,
            "bass": content["bass"],
            "rhythm": None  # Rhythm generation not yet implemented
        }

    # Step 4: Orchestration - assign instruments to content
    print(f"\n[ORCHESTRATOR] Step 4: Invoking Orchestration Agent...")
    orchestration = await invoke_orchestration_agent(style_guide, sections, all_content)

    # Step 5: Compile to MIDI tool calls
    print(f"[ORCHESTRATOR] Step 5: Compiling to MIDI tool calls...")
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
            print(f"[ORCHESTRATOR]   Created track {track_id}: {track_name}")

    # Compile content with orchestration
    for section_name, content in all_content.items():
        print(f"[ORCHESTRATOR]   Compiling section: {section_name}")

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

    print(f"[ORCHESTRATOR] Orchestration complete! Generated {len(all_tool_calls)} tool calls")

    return {
        "style_guide": style_guide,
        "sections": sections,
        "tool_calls": all_tool_calls
    }


async def orchestrate_composition_with_revision(
    user_prompt: str,
    target_bars: int = 32,
    max_revision_cycles: int = 3
) -> Dict:
    """Orchestrate with quality validation and revision.

    Adds:
    - Critic Agent evaluation after each section
    - Revision loop for quality control
    - Detailed reporting on musical quality

    The revision loop ensures that generated music meets quality thresholds
    before compilation to MIDI.

    Args:
        user_prompt: User's description of desired composition
        target_bars: Target length in bars (default 32)
        max_revision_cycles: Maximum number of revision attempts per section (default 3)

    Returns:
        Dictionary with:
        - style_guide: StyleGuide object
        - sections: List of Section objects
        - tool_calls: List of tool call dictionaries for frontend
        - critiques: List of CriticReport objects (one per section)
        - total_revision_cycles: Total number of revisions performed
    """
    print(f"[ORCHESTRATOR] Composition with revision loop (max {max_revision_cycles} cycles)")

    # Step 1: Style & Structure
    print(f"[ORCHESTRATOR] Step 1: Invoking Style Agent...")
    style_guide = await invoke_style_agent(user_prompt)
    print(f"[ORCHESTRATOR] Style: {style_guide.genre} {style_guide.subgenre}")

    print(f"[ORCHESTRATOR] Step 2: Invoking Arranger Agent...")
    sections = await invoke_arranger_agent(style_guide, target_bars)
    print(f"[ORCHESTRATOR] Created {len(sections)} sections")

    # Step 2: Content generation with revision loop
    all_compositions = {}  # section_name → {harmony, melody, rhythm, critique}
    all_critiques = []
    total_revision_cycles = 0

    for i, section in enumerate(sections):
        print(f"\n[ORCHESTRATOR] Section {i+1}/{len(sections)}: {section.name} (bars {section.bars[0]}-{section.bars[1]})")

        revision_cycle = 0
        best_composition = None
        best_critique = None

        while revision_cycle < max_revision_cycles:
            print(f"[ORCHESTRATOR]   Revision cycle {revision_cycle + 1}/{max_revision_cycles}")

            # Generate harmony, melody, and bass together
            # Note: Using generic placeholders - instruments will be assigned by orchestration agent
            print(f"[ORCHESTRATOR]   - Generating musical content (harmony, melody, bass)...")
            content = await invoke_musical_content_agent(style_guide, section, harmony_track="harmony_track", melody_track="melody_track", bass_track="bass_track")
            harmony = content["harmony"]
            melody = content["melody"] if section.name not in ["intro", "outro"] else None
            bass = content["bass"]

            # Generate rhythm - TODO: Implement rhythm generation or use a simple pattern
            # For now, creating a None placeholder as rhythm_agent was removed
            print(f"[ORCHESTRATOR]   - Skipping rhythm (rhythm_agent removed)")
            rhythm = None

            # Evaluate with Critic
            print(f"[ORCHESTRATOR]   - Evaluating with Critic Agent...")
            critique = await invoke_critic_agent(
                style_guide=style_guide,
                harmony=harmony,
                melody=melody,
                rhythm=rhythm
            )

            # Store best attempt (highest score)
            if best_critique is None or critique.overall_score > best_critique.overall_score:
                best_composition = {
                    "harmony": harmony,
                    "melody": melody,
                    "bass": bass,
                    "rhythm": rhythm
                }
                best_critique = critique

            if critique.passed:
                # Quality threshold met!
                print(f"[ORCHESTRATOR]   ✓ Section {section.name} PASSED critique (score: {critique.overall_score:.2f})")
                all_compositions[section.name] = best_composition
                all_critiques.append(critique)
                break
            else:
                # Log issues and retry
                print(f"[ORCHESTRATOR]   ✗ Section {section.name} failed critique (score: {critique.overall_score:.2f})")
                print(f"[ORCHESTRATOR]     Errors: {sum(1 for i in critique.issues if i.get('severity') == 'error')}, "
                      f"Warnings: {sum(1 for i in critique.issues if i.get('severity') == 'warning')}")
                print(f"[ORCHESTRATOR]     Agents needing revision: {', '.join(critique.revision_needed)}")

                revision_cycle += 1
                total_revision_cycles += 1

        # If we exceeded max cycles, accept the best attempt
        if revision_cycle >= max_revision_cycles:
            print(f"[ORCHESTRATOR]   ⚠ Section {section.name} did not pass after {max_revision_cycles} cycles")
            print(f"[ORCHESTRATOR]     Accepting best attempt (score: {best_critique.overall_score:.2f})")
            all_compositions[section.name] = best_composition
            all_critiques.append(best_critique)

    # Step 3: Compile to tool calls
    print(f"\n[ORCHESTRATOR] Compiling to MIDI tool calls...")
    all_tool_calls = []

    # Create tracks
    piano_track_id = 1
    melody_track_id = 2
    drums_track_id = 3

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

    # Compile each section's content
    for section_name, content in all_compositions.items():
        # Compile harmony
        harmony_calls = compile_progression_to_tool_calls(content["harmony"], piano_track_id)
        all_tool_calls.extend(harmony_calls)

        # Compile melody
        if content["melody"]:
            melody_notes = compile_melody_to_notes(content["melody"])
            all_tool_calls.append({
                "name": "addNotes",
                "args": {"trackId": melody_track_id, "notes": melody_notes}
            })

        # Compile rhythm (skip if None)
        if content.get("rhythm") is not None:
            drum_notes = compile_drums_to_notes(content["rhythm"])
            all_tool_calls.append({
                "name": "addNotes",
                "args": {"trackId": drums_track_id, "notes": drum_notes}
            })

    # Calculate overall quality metrics
    avg_score = sum(c.overall_score for c in all_critiques) / len(all_critiques) if all_critiques else 0.0
    passed_count = sum(1 for c in all_critiques if c.passed)

    print(f"\n[ORCHESTRATOR] ✓ Orchestration complete!")
    print(f"[ORCHESTRATOR]   - Generated {len(all_tool_calls)} tool calls")
    print(f"[ORCHESTRATOR]   - Average quality score: {avg_score:.2f}/1.00")
    print(f"[ORCHESTRATOR]   - Sections passed: {passed_count}/{len(all_critiques)}")
    print(f"[ORCHESTRATOR]   - Total revision cycles: {total_revision_cycles}")

    return {
        "style_guide": style_guide,
        "sections": sections,
        "tool_calls": all_tool_calls,
        "critiques": all_critiques,
        "total_revision_cycles": total_revision_cycles,
        "average_score": avg_score
    }


async def orchestrate_full_composition(
    user_prompt: str,
    target_bars: int = 32,
    max_revision_cycles: int = 3
) -> Dict:
    """Complete multi-agent composition pipeline.

    Full multi-agent system with:
    - Bass Agent for foundational bass lines
    - Orchestration Agent for instrument assignment and dynamics
    - All agents (Style, Arranger, Harmony, Melody, Rhythm, Critic)

    Args:
        user_prompt: User's description of desired composition
        target_bars: Target length in bars (default 32)
        max_revision_cycles: Maximum number of revision attempts per section (default 3)

    Returns:
        Dictionary with:
        - style_guide: StyleGuide object
        - sections: List of Section objects
        - orchestration: OrchestrationPlan object
        - tool_calls: List of tool call dictionaries for frontend
        - critiques: List of CriticReport objects
        - total_revision_cycles: Total number of revisions performed
        - average_score: Average quality score
    """
    print(f"[ORCHESTRATOR] Full multi-agent composition pipeline")

    # Step 1: Planning
    print(f"[ORCHESTRATOR] Step 1: Invoking Style Agent...")
    style_guide = await invoke_style_agent(user_prompt)
    print(f"[ORCHESTRATOR] Style: {style_guide.genre} {style_guide.subgenre}")

    print(f"[ORCHESTRATOR] Step 2: Invoking Arranger Agent...")
    sections = await invoke_arranger_agent(style_guide, target_bars)
    print(f"[ORCHESTRATOR] Created {len(sections)} sections")

    # Step 2: Content generation per section (with revision)
    all_content = {}  # section_name → {harmony, melody, bass, rhythm}
    all_critiques = []
    total_revision_cycles = 0

    for i, section in enumerate(sections):
        print(f"\n[ORCHESTRATOR] Section {i+1}/{len(sections)}: {section.name} (bars {section.bars[0]}-{section.bars[1]})")

        revision_cycle = 0
        best_composition = None
        best_critique = None

        while revision_cycle < max_revision_cycles:
            print(f"[ORCHESTRATOR]   Revision cycle {revision_cycle + 1}/{max_revision_cycles}")

            # Generate all content together
            # Note: Using generic placeholders - orchestration agent will assign actual instruments
            print(f"[ORCHESTRATOR]   - Generating musical content (harmony, melody, bass)...")
            content = await invoke_musical_content_agent(style_guide, section, harmony_track="harmony_track", melody_track="melody_track", bass_track="bass_track")
            harmony = content["harmony"]
            melody = content["melody"] if section.name not in ["intro", "outro"] else None
            bass = content["bass"]

            # Rhythm - TODO: Implement rhythm generation or use a simple pattern
            # For now, creating a None placeholder as rhythm_agent was removed
            print(f"[ORCHESTRATOR]   - Skipping rhythm (rhythm_agent removed)")
            rhythm = None

            # Critique
            print(f"[ORCHESTRATOR]   - Evaluating with Critic Agent...")
            critique = await invoke_critic_agent(
                style_guide=style_guide,
                harmony=harmony,
                melody=melody,
                rhythm=rhythm
                # Note: bass not critiqued yet
            )

            # Store best attempt
            if best_critique is None or critique.overall_score > best_critique.overall_score:
                best_composition = {
                    "harmony": harmony,
                    "melody": melody,
                    "bass": bass,
                    "rhythm": rhythm
                }
                best_critique = critique

            if critique.passed:
                print(f"[ORCHESTRATOR]   ✓ Section {section.name} PASSED critique (score: {critique.overall_score:.2f})")
                all_content[section.name] = best_composition
                all_critiques.append(critique)
                break
            else:
                print(f"[ORCHESTRATOR]   ✗ Section {section.name} failed critique (score: {critique.overall_score:.2f})")
                print(f"[ORCHESTRATOR]     Errors: {sum(1 for i in critique.issues if i.get('severity') == 'error')}, "
                      f"Warnings: {sum(1 for i in critique.issues if i.get('severity') == 'warning')}")
                revision_cycle += 1
                total_revision_cycles += 1

        # Accept best attempt if exceeded max cycles
        if revision_cycle >= max_revision_cycles:
            print(f"[ORCHESTRATOR]   ⚠ Section {section.name} did not pass after {max_revision_cycles} cycles")
            print(f"[ORCHESTRATOR]     Accepting best attempt (score: {best_critique.overall_score:.2f})")
            all_content[section.name] = best_composition
            all_critiques.append(best_critique)

    # Step 3: Orchestration - assigns to instruments, dynamics
    print(f"\n[ORCHESTRATOR] Step 3: Invoking Orchestration Agent...")
    orchestration = await invoke_orchestration_agent(style_guide, sections, all_content)

    # Step 4: Compilation to MIDI tool calls
    print(f"[ORCHESTRATOR] Step 4: Compiling to MIDI tool calls...")
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
            print(f"[ORCHESTRATOR]   Created track {track_id}: {track_name}")

    # Compile content with orchestration
    for section_name, content in all_content.items():
        print(f"[ORCHESTRATOR]   Compiling section: {section_name}")

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

    # Calculate overall quality metrics
    avg_score = sum(c.overall_score for c in all_critiques) / len(all_critiques) if all_critiques else 0.0
    passed_count = sum(1 for c in all_critiques if c.passed)

    print(f"\n[ORCHESTRATOR] ✓ Full composition complete!")
    print(f"[ORCHESTRATOR]   - Generated {len(all_tool_calls)} tool calls")
    print(f"[ORCHESTRATOR]   - Created {len(track_map)} tracks: {', '.join(orchestration.track_order)}")
    print(f"[ORCHESTRATOR]   - Average quality score: {avg_score:.2f}/1.00")
    print(f"[ORCHESTRATOR]   - Sections passed: {passed_count}/{len(all_critiques)}")
    print(f"[ORCHESTRATOR]   - Total revision cycles: {total_revision_cycles}")

    return {
        "style_guide": style_guide,
        "sections": sections,
        "orchestration": orchestration,
        "tool_calls": all_tool_calls,
        "critiques": all_critiques,
        "total_revision_cycles": total_revision_cycles,
        "average_score": avg_score
    }
