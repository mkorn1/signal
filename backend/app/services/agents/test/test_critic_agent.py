"""Unit tests for Critic Agent."""

import pytest
from app.services.mir.schema import (
    Chord, ChordProgression, Note, MelodyPhrase, StyleGuide, Section
)
from app.services.agents.critic_agent import (
    invoke_critic_agent,
    get_issues_for_agent,
    format_critique_for_agent,
    PASS_THRESHOLD
)


class TestCriticAgent:
    """Tests for invoke_critic_agent function."""

    @pytest.mark.asyncio
    async def test_passes_clean_composition(self):
        """Test that a composition with no issues passes."""
        style_guide = StyleGuide(
            genre="jazz",
            subgenre="ballad",
            harmonic_complexity="complex",
            swing=0.55,
            extensions_allowed=["7", "9", "11", "13"],
            tempo_range=(60, 90)
        )

        # Create clean progression with good voice leading
        harmony = ChordProgression(
            track="piano",
            section="verse",
            chords=[
                Chord(root="D", quality="m7", bar=1, beat=1.0, duration="whole",
                      voicing=["D2", "A2", "F3", "C4"], velocity=75),
                Chord(root="G", quality="7", bar=2, beat=1.0, duration="whole",
                      voicing=["G2", "B2", "F3", "D4"], velocity=75),
                Chord(root="C", quality="maj7", bar=3, beat=1.0, duration="whole",
                      voicing=["C2", "B2", "E3", "C4"], velocity=75),
            ]
        )

        melody = MelodyPhrase(
            track="flute",
            section="verse",
            notes=[
                Note(pitch="D4", bar=1, beat=1.0, duration="quarter", velocity=80),
                Note(pitch="F4", bar=1, beat=2.0, duration="quarter", velocity=80),
                Note(pitch="E4", bar=1, beat=3.0, duration="quarter", velocity=80),
            ]
        )

        critique = await invoke_critic_agent(style_guide, harmony, melody)

        assert critique.passed is True, "Clean composition should pass"
        assert critique.overall_score >= PASS_THRESHOLD, f"Score should be >= {PASS_THRESHOLD}"
        assert len(critique.issues) == 0, "Should have no issues"
        assert len(critique.revision_needed) == 0, "Should need no revisions"

    @pytest.mark.asyncio
    async def test_fails_with_parallel_fifths(self):
        """Test that parallel fifths cause failure."""
        style_guide = StyleGuide(
            genre="jazz",
            subgenre="ballad",
            harmonic_complexity="complex",
            swing=0.55,
            extensions_allowed=["7", "9", "11", "13"],
            tempo_range=(60, 90)
        )

        # Create progression with parallel fifths
        harmony = ChordProgression(
            track="piano",
            section="verse",
            chords=[
                Chord(root="C", quality="maj", bar=1, beat=1.0, duration="whole",
                      voicing=["C3", "G3", "C4"], velocity=75),
                Chord(root="D", quality="maj", bar=2, beat=1.0, duration="whole",
                      voicing=["D3", "A3", "D4"], velocity=75),  # Parallel motion
            ]
        )

        critique = await invoke_critic_agent(style_guide, harmony)

        assert critique.passed is False, "Should fail with parallel fifths"
        assert critique.overall_score < PASS_THRESHOLD, "Score should be below threshold"
        assert len(critique.issues) > 0, "Should have issues"
        assert "harmony" in critique.revision_needed, "Harmony agent should need revision"

    @pytest.mark.asyncio
    async def test_fails_with_range_violation(self):
        """Test that melody range violations cause failure."""
        style_guide = StyleGuide(
            genre="pop",
            subgenre="simple",
            harmonic_complexity="simple",
            swing=0.0,
            extensions_allowed=["7"],
            tempo_range=(100, 130)
        )

        harmony = ChordProgression(
            track="piano",
            section="verse",
            chords=[
                Chord(root="C", quality="maj", bar=1, beat=1.0, duration="whole",
                      voicing=["C2", "E3", "G3"], velocity=75),
            ]
        )

        # Melody with out-of-range note
        melody = MelodyPhrase(
            track="flute",
            section="verse",
            notes=[
                Note(pitch="A5", bar=1, beat=1.0, duration="quarter", velocity=80),  # Above G5
            ]
        )

        critique = await invoke_critic_agent(style_guide, harmony, melody)

        assert critique.passed is False, "Should fail with range violation"
        assert len(critique.issues) > 0, "Should have issues"
        range_issues = [i for i in critique.issues if i["type"] == "range_violation"]
        assert len(range_issues) > 0, "Should have range violation issue"
        assert "melody" in critique.revision_needed, "Melody agent should need revision"

    @pytest.mark.asyncio
    async def test_warns_on_style_violation(self):
        """Test that style violations generate warnings."""
        style_guide = StyleGuide(
            genre="pop",
            subgenre="simple",
            harmonic_complexity="simple",
            swing=0.0,
            extensions_allowed=["7"],  # Only 7ths allowed
            tempo_range=(100, 130)
        )

        # Use complex jazz extensions (not allowed in simple pop)
        harmony = ChordProgression(
            track="piano",
            section="verse",
            chords=[
                Chord(root="C", quality="maj", bar=1, beat=1.0, duration="whole",
                      voicing=["C2", "E3", "G3"], velocity=75),
                Chord(root="D", quality="m9", bar=2, beat=1.0, duration="whole",
                      voicing=["D2", "A2", "F3", "E4"], velocity=75),  # 9th not allowed
            ]
        )

        critique = await invoke_critic_agent(style_guide, harmony)

        # Warnings reduce score but may still pass if score is high enough
        style_issues = [i for i in critique.issues if i["type"] == "style_violation"]
        assert len(style_issues) > 0, "Should have style violation warnings"
        assert all(i["severity"] == "warning" for i in style_issues)

    @pytest.mark.asyncio
    async def test_score_calculation(self):
        """Test that score is calculated correctly based on errors and warnings."""
        style_guide = StyleGuide(
            genre="jazz",
            subgenre="ballad",
            harmonic_complexity="complex",
            swing=0.55,
            extensions_allowed=["7", "9"],
            tempo_range=(60, 90)
        )

        # Create progression with 1 error (parallel fifths)
        harmony = ChordProgression(
            track="piano",
            section="verse",
            chords=[
                Chord(root="C", quality="maj", bar=1, beat=1.0, duration="whole",
                      voicing=["C3", "G3", "C4"], velocity=75),
                Chord(root="D", quality="maj", bar=2, beat=1.0, duration="whole",
                      voicing=["D3", "A3", "D4"], velocity=75),
            ]
        )

        critique = await invoke_critic_agent(style_guide, harmony)

        error_count = sum(1 for i in critique.issues if i["severity"] == "error")
        warning_count = sum(1 for i in critique.issues if i["severity"] == "warning")

        # Score should be 1.0 - (errors * 0.2) - (warnings * 0.05)
        expected_max_score = 1.0 - (error_count * 0.2) - (warning_count * 0.05)

        assert critique.overall_score <= expected_max_score, "Score should match penalty formula"

    @pytest.mark.asyncio
    async def test_handles_none_melody(self):
        """Test that critic handles None melody (intro/outro sections)."""
        style_guide = StyleGuide(
            genre="jazz",
            subgenre="ballad",
            harmonic_complexity="complex",
            swing=0.55,
            extensions_allowed=["7", "9", "11", "13"],
            tempo_range=(60, 90)
        )

        harmony = ChordProgression(
            track="piano",
            section="intro",
            chords=[
                Chord(root="D", quality="m7", bar=1, beat=1.0, duration="whole",
                      voicing=["D2", "A2", "F3", "C4"], velocity=75),
            ]
        )

        # Pass None for melody
        critique = await invoke_critic_agent(style_guide, harmony, melody=None)

        # Should complete without errors
        assert critique is not None
        assert isinstance(critique.overall_score, float)


class TestCriticHelpers:
    """Tests for critic helper functions."""

    @pytest.mark.asyncio
    async def test_get_issues_for_agent(self):
        """Test filtering issues by agent."""
        style_guide = StyleGuide(
            genre="jazz",
            subgenre="ballad",
            harmonic_complexity="complex",
            swing=0.55,
            extensions_allowed=["7", "9"],
            tempo_range=(60, 90)
        )

        # Create composition with issues from both harmony and melody
        harmony = ChordProgression(
            track="piano",
            section="verse",
            chords=[
                Chord(root="C", quality="maj", bar=1, beat=1.0, duration="whole",
                      voicing=["C3", "G3", "C4"], velocity=75),
                Chord(root="D", quality="maj", bar=2, beat=1.0, duration="whole",
                      voicing=["D3", "A3", "D4"], velocity=75),
            ]
        )

        melody = MelodyPhrase(
            track="flute",
            section="verse",
            notes=[
                Note(pitch="A5", bar=1, beat=1.0, duration="quarter", velocity=80),
            ]
        )

        critique = await invoke_critic_agent(style_guide, harmony, melody)

        harmony_issues = get_issues_for_agent(critique, "harmony")
        melody_issues = get_issues_for_agent(critique, "melody")

        assert len(harmony_issues) > 0, "Should have harmony issues"
        assert len(melody_issues) > 0, "Should have melody issues"
        assert all(i["agent"] == "harmony" for i in harmony_issues)
        assert all(i["agent"] == "melody" for i in melody_issues)

    @pytest.mark.asyncio
    async def test_format_critique_for_agent(self):
        """Test formatting critique feedback for agents."""
        style_guide = StyleGuide(
            genre="jazz",
            subgenre="ballad",
            harmonic_complexity="complex",
            swing=0.55,
            extensions_allowed=["7", "9"],
            tempo_range=(60, 90)
        )

        harmony = ChordProgression(
            track="piano",
            section="verse",
            chords=[
                Chord(root="C", quality="maj", bar=1, beat=1.0, duration="whole",
                      voicing=["C3", "G3", "C4"], velocity=75),
                Chord(root="D", quality="maj", bar=2, beat=1.0, duration="whole",
                      voicing=["D3", "A3", "D4"], velocity=75),
            ]
        )

        critique = await invoke_critic_agent(style_guide, harmony)

        feedback = format_critique_for_agent(critique, "harmony")

        assert isinstance(feedback, str)
        assert "harmony" in feedback.lower()
        # Should contain issue information
        if len(get_issues_for_agent(critique, "harmony")) > 0:
            assert "Location:" in feedback
            assert "Message:" in feedback
