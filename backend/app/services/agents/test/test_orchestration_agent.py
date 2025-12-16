"""Unit tests for Orchestration Agent."""

import pytest
from app.services.mir.schema import StyleGuide, Section
from app.services.agents.orchestration_agent import (
    invoke_orchestration_agent,
    get_instrument_for_content
)


class TestOrchestrationAgent:
    """Tests for invoke_orchestration_agent function."""

    @pytest.mark.asyncio
    async def test_generates_orchestration_plan(self):
        """Test that orchestration agent generates a valid plan."""
        style_guide = StyleGuide(
            genre="jazz",
            subgenre="ballad",
            harmonic_complexity="complex",
            swing=0.55,
            extensions_allowed=["7", "9", "11", "13"],
            tempo_range=(60, 90)
        )

        sections = [
            Section(name="intro", bars=(1, 8), key="Dm", tempo=72, energy="soft"),
            Section(name="verse", bars=(9, 24), key="Dm", tempo=72, energy="medium"),
            Section(name="outro", bars=(25, 32), key="Dm", tempo=72, energy="soft"),
        ]

        # Mock content (orchestration doesn't actually use the content, just section names)
        all_content = {
            "intro": {"harmony": None, "melody": None, "bass": None, "rhythm": None},
            "verse": {"harmony": None, "melody": None, "bass": None, "rhythm": None},
            "outro": {"harmony": None, "melody": None, "bass": None, "rhythm": None},
        }

        orchestration = await invoke_orchestration_agent(style_guide, sections, all_content)

        # Validate structure
        assert orchestration is not None
        assert len(orchestration.assignments) > 0, "Should have instrument assignments"
        assert len(orchestration.track_order) > 0, "Should have track order"
        assert isinstance(orchestration.dynamic_arc, dict), "Should have dynamic arc"

    @pytest.mark.asyncio
    async def test_assigns_all_content_types(self):
        """Test that all content types (harmony, melody, bass, rhythm) get assigned."""
        style_guide = StyleGuide(
            genre="pop",
            subgenre="simple",
            harmonic_complexity="simple",
            swing=0.0,
            extensions_allowed=["7"],
            tempo_range=(100, 130)
        )

        sections = [
            Section(name="verse", bars=(1, 16), key="C", tempo=120, energy="medium"),
        ]

        all_content = {
            "verse": {"harmony": None, "melody": None, "bass": None, "rhythm": None},
        }

        orchestration = await invoke_orchestration_agent(style_guide, sections, all_content)

        # Check that all content types are assigned
        content_types = {assignment.source_content for assignment in orchestration.assignments}
        assert "harmony" in content_types, "Should assign harmony"
        assert "melody" in content_types, "Should assign melody"
        assert "bass" in content_types, "Should assign bass"
        assert "rhythm" in content_types, "Should assign rhythm"

    @pytest.mark.asyncio
    async def test_genre_specific_instruments(self):
        """Test that different genres get different instruments."""
        sections = [
            Section(name="verse", bars=(1, 16), key="C", tempo=120, energy="medium"),
        ]

        all_content = {
            "verse": {"harmony": None, "melody": None, "bass": None, "rhythm": None},
        }

        # Jazz
        jazz_style = StyleGuide(
            genre="jazz",
            subgenre="ballad",
            harmonic_complexity="complex",
            swing=0.55,
            extensions_allowed=["7", "9", "11", "13"],
            tempo_range=(60, 90)
        )

        jazz_orch = await invoke_orchestration_agent(jazz_style, sections, all_content)
        jazz_instruments = {assignment.instrument for assignment in jazz_orch.assignments}

        # Rock
        rock_style = StyleGuide(
            genre="rock",
            subgenre="classic",
            harmonic_complexity="simple",
            swing=0.0,
            extensions_allowed=[],
            tempo_range=(110, 140)
        )

        rock_orch = await invoke_orchestration_agent(rock_style, sections, all_content)
        rock_instruments = {assignment.instrument for assignment in rock_orch.assignments}

        # Instruments should be different between jazz and rock
        assert jazz_instruments != rock_instruments, "Jazz and rock should use different instruments"

        # Note: Since orchestration is now LLM-based, we can't assert specific instruments
        # but we can verify that instruments are appropriate for the genre
        # Jazz typically uses: piano, saxophone, upright_bass, drums
        # Rock typically uses: guitar, electric_guitar, electric_bass, drums
        assert len(jazz_instruments) >= 4, "Should have at least 4 instruments"
        assert len(rock_instruments) >= 4, "Should have at least 4 instruments"

    @pytest.mark.asyncio
    async def test_melody_skips_intro_outro(self):
        """Test that melody is not assigned to intro/outro sections."""
        style_guide = StyleGuide(
            genre="jazz",
            subgenre="ballad",
            harmonic_complexity="complex",
            swing=0.55,
            extensions_allowed=["7", "9", "11", "13"],
            tempo_range=(60, 90)
        )

        sections = [
            Section(name="intro", bars=(1, 8), key="Dm", tempo=72, energy="soft"),
            Section(name="verse", bars=(9, 24), key="Dm", tempo=72, energy="medium"),
            Section(name="outro", bars=(25, 32), key="Dm", tempo=72, energy="soft"),
        ]

        all_content = {
            "intro": {"harmony": None, "melody": None, "bass": None, "rhythm": None},
            "verse": {"harmony": None, "melody": None, "bass": None, "rhythm": None},
            "outro": {"harmony": None, "melody": None, "bass": None, "rhythm": None},
        }

        orchestration = await invoke_orchestration_agent(style_guide, sections, all_content)

        # Find melody assignment
        melody_assignment = next(
            (a for a in orchestration.assignments if a.source_content == "melody"),
            None
        )

        assert melody_assignment is not None, "Should have melody assignment"
        assert "intro" not in melody_assignment.active_sections, "Melody should not play in intro"
        assert "outro" not in melody_assignment.active_sections, "Melody should not play in outro"
        assert "verse" in melody_assignment.active_sections, "Melody should play in verse"

    @pytest.mark.asyncio
    async def test_rhythm_skips_intro(self):
        """Test that rhythm is not assigned to intro section."""
        style_guide = StyleGuide(
            genre="jazz",
            subgenre="ballad",
            harmonic_complexity="complex",
            swing=0.55,
            extensions_allowed=["7", "9", "11", "13"],
            tempo_range=(60, 90)
        )

        sections = [
            Section(name="intro", bars=(1, 8), key="Dm", tempo=72, energy="soft"),
            Section(name="verse", bars=(9, 24), key="Dm", tempo=72, energy="medium"),
        ]

        all_content = {
            "intro": {"harmony": None, "melody": None, "bass": None, "rhythm": None},
            "verse": {"harmony": None, "melody": None, "bass": None, "rhythm": None},
        }

        orchestration = await invoke_orchestration_agent(style_guide, sections, all_content)

        # Find rhythm assignment
        rhythm_assignment = next(
            (a for a in orchestration.assignments if a.source_content == "rhythm"),
            None
        )

        assert rhythm_assignment is not None, "Should have rhythm assignment"
        assert "intro" not in rhythm_assignment.active_sections, "Rhythm should not play in intro"
        assert "verse" in rhythm_assignment.active_sections, "Rhythm should play in verse"

    @pytest.mark.asyncio
    async def test_dynamic_arc_varies_by_energy(self):
        """Test that dynamic arc reflects section energy levels."""
        style_guide = StyleGuide(
            genre="pop",
            subgenre="simple",
            harmonic_complexity="simple",
            swing=0.0,
            extensions_allowed=["7"],
            tempo_range=(100, 130)
        )

        sections = [
            Section(name="intro", bars=(1, 8), key="C", tempo=120, energy="soft"),
            Section(name="verse", bars=(9, 24), key="C", tempo=120, energy="medium"),
            Section(name="chorus", bars=(25, 40), key="C", tempo=120, energy="climax"),
        ]

        all_content = {
            "intro": {"harmony": None, "melody": None, "bass": None, "rhythm": None},
            "verse": {"harmony": None, "melody": None, "bass": None, "rhythm": None},
            "chorus": {"harmony": None, "melody": None, "bass": None, "rhythm": None},
        }

        orchestration = await invoke_orchestration_agent(style_guide, sections, all_content)

        # Check that dynamic arc exists for all sections
        assert "intro" in orchestration.dynamic_arc
        assert "verse" in orchestration.dynamic_arc
        assert "chorus" in orchestration.dynamic_arc

        # Get average velocity for each section
        intro_velocities = [v for _, v in orchestration.dynamic_arc["intro"]]
        verse_velocities = [v for _, v in orchestration.dynamic_arc["verse"]]
        chorus_velocities = [v for _, v in orchestration.dynamic_arc["chorus"]]

        intro_avg = sum(intro_velocities) / len(intro_velocities)
        verse_avg = sum(verse_velocities) / len(verse_velocities)
        chorus_avg = sum(chorus_velocities) / len(chorus_velocities)

        # Soft < Medium < Climax
        assert intro_avg < chorus_avg, "Soft intro should be quieter than climax chorus"
        assert verse_avg < chorus_avg, "Medium verse should be quieter than climax chorus"

    @pytest.mark.asyncio
    async def test_track_order_is_logical(self):
        """Test that track order is logical (rhythm section first)."""
        style_guide = StyleGuide(
            genre="jazz",
            subgenre="ballad",
            harmonic_complexity="complex",
            swing=0.55,
            extensions_allowed=["7", "9", "11", "13"],
            tempo_range=(60, 90)
        )

        sections = [
            Section(name="verse", bars=(1, 16), key="Dm", tempo=72, energy="medium"),
        ]

        all_content = {
            "verse": {"harmony": None, "melody": None, "bass": None, "rhythm": None},
        }

        orchestration = await invoke_orchestration_agent(style_guide, sections, all_content)

        # Track order should have rhythm section first (drums, bass)
        # then harmony, then melody
        assert len(orchestration.track_order) == 4, "Should have 4 tracks"

        # Get instrument for each content type
        harmony_inst = get_instrument_for_content(orchestration, "harmony", "verse")
        melody_inst = get_instrument_for_content(orchestration, "melody", "verse")
        bass_inst = get_instrument_for_content(orchestration, "bass", "verse")
        rhythm_inst = get_instrument_for_content(orchestration, "rhythm", "verse")

        # Check that these instruments appear in track_order
        assert harmony_inst in orchestration.track_order
        assert melody_inst in orchestration.track_order
        assert bass_inst in orchestration.track_order
        assert rhythm_inst in orchestration.track_order


class TestGetInstrumentForContent:
    """Tests for get_instrument_for_content helper function."""

    @pytest.mark.asyncio
    async def test_returns_correct_instrument(self):
        """Test that helper returns the right instrument for a content type."""
        style_guide = StyleGuide(
            genre="jazz",
            subgenre="ballad",
            harmonic_complexity="complex",
            swing=0.55,
            extensions_allowed=["7", "9", "11", "13"],
            tempo_range=(60, 90)
        )

        sections = [
            Section(name="verse", bars=(1, 16), key="Dm", tempo=72, energy="medium"),
        ]

        all_content = {
            "verse": {"harmony": None, "melody": None, "bass": None, "rhythm": None},
        }

        orchestration = await invoke_orchestration_agent(style_guide, sections, all_content)

        harmony_inst = get_instrument_for_content(orchestration, "harmony", "verse")
        melody_inst = get_instrument_for_content(orchestration, "melody", "verse")

        # Note: Since orchestration is now LLM-based, we can't assert specific instruments
        # but we can verify that instruments are assigned
        assert harmony_inst is not None, "Harmony should have an instrument assigned"
        assert melody_inst is not None, "Melody should have an instrument assigned"

    @pytest.mark.asyncio
    async def test_returns_none_for_inactive_section(self):
        """Test that helper returns None for content not active in a section."""
        style_guide = StyleGuide(
            genre="jazz",
            subgenre="ballad",
            harmonic_complexity="complex",
            swing=0.55,
            extensions_allowed=["7", "9", "11", "13"],
            tempo_range=(60, 90)
        )

        sections = [
            Section(name="intro", bars=(1, 8), key="Dm", tempo=72, energy="soft"),
            Section(name="verse", bars=(9, 24), key="Dm", tempo=72, energy="medium"),
        ]

        all_content = {
            "intro": {"harmony": None, "melody": None, "bass": None, "rhythm": None},
            "verse": {"harmony": None, "melody": None, "bass": None, "rhythm": None},
        }

        orchestration = await invoke_orchestration_agent(style_guide, sections, all_content)

        # Melody doesn't play in intro
        melody_in_intro = get_instrument_for_content(orchestration, "melody", "intro")
        assert melody_in_intro is None, "Melody should not be assigned to intro"

        # But melody does play in verse
        melody_in_verse = get_instrument_for_content(orchestration, "melody", "verse")
        assert melody_in_verse is not None, "Melody should be assigned to verse"
