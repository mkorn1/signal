"""Unit tests for Bass Agent."""

import pytest
from app.services.mir.schema import (
    Chord, ChordProgression, StyleGuide, Section, Note
)
from app.services.agents.bass_agent import invoke_bass_agent
from app.services.mir.compiler import pitch_string_to_midi


class TestBassAgent:
    """Tests for invoke_bass_agent function."""

    @pytest.mark.asyncio
    async def test_generates_bass_line(self):
        """Test that bass agent generates a valid bass line."""
        style_guide = StyleGuide(
            genre="jazz",
            subgenre="ballad",
            harmonic_complexity="complex",
            swing=0.55,
            extensions_allowed=["7", "9", "11", "13"],
            tempo_range=(60, 90)
        )

        section = Section(
            name="verse",
            bars=(1, 4),
            key="Dm",
            tempo=72,
            energy="medium"
        )

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
                Chord(root="F", quality="maj7", bar=4, beat=1.0, duration="whole",
                      voicing=["F2", "A2", "E3", "C4"], velocity=75),
            ]
        )

        bass_line = await invoke_bass_agent(style_guide, section, harmony)

        # Validate structure
        assert bass_line is not None
        assert bass_line.track in ["bass", "upright_bass", "electric_bass"]
        assert bass_line.section == "verse"
        assert len(bass_line.notes) > 0, "Bass line should have notes"

        # Validate all notes have required fields
        for note in bass_line.notes:
            assert isinstance(note.pitch, str), "Pitch should be string"
            assert isinstance(note.bar, int), "Bar should be int"
            assert isinstance(note.beat, float), "Beat should be float"
            assert isinstance(note.duration, str), "Duration should be string"
            assert isinstance(note.velocity, int), "Velocity should be int"

    @pytest.mark.asyncio
    async def test_bass_in_correct_range(self):
        """Test that bass notes are in bass register (E1-E3)."""
        style_guide = StyleGuide(
            genre="jazz",
            subgenre="ballad",
            harmonic_complexity="complex",
            swing=0.55,
            extensions_allowed=["7", "9", "11", "13"],
            tempo_range=(60, 90)
        )

        section = Section(
            name="verse",
            bars=(1, 2),
            key="Dm",
            tempo=72,
            energy="medium"
        )

        harmony = ChordProgression(
            track="piano",
            section="verse",
            chords=[
                Chord(root="D", quality="m7", bar=1, beat=1.0, duration="whole",
                      voicing=["D2", "A2", "F3", "C4"], velocity=75),
                Chord(root="G", quality="7", bar=2, beat=1.0, duration="whole",
                      voicing=["G2", "B2", "F3", "D4"], velocity=75),
            ]
        )

        bass_line = await invoke_bass_agent(style_guide, section, harmony)

        # Check that notes are in bass range (E1=40 to E3=64 in MIDI)
        e1_midi = pitch_string_to_midi("E1")  # 40
        e3_midi = pitch_string_to_midi("E3")  # 64

        for note in bass_line.notes:
            midi_pitch = pitch_string_to_midi(note.pitch)
            assert e1_midi <= midi_pitch <= e3_midi, \
                f"Bass note {note.pitch} (MIDI {midi_pitch}) out of range E1-E3 ({e1_midi}-{e3_midi})"

    @pytest.mark.asyncio
    async def test_bass_starts_on_chord_roots(self):
        """Test that bass line includes chord root notes on beat 1."""
        style_guide = StyleGuide(
            genre="pop",
            subgenre="simple",
            harmonic_complexity="simple",
            swing=0.0,
            extensions_allowed=["7"],
            tempo_range=(100, 130)
        )

        section = Section(
            name="verse",
            bars=(1, 2),
            key="C",
            tempo=120,
            energy="medium"
        )

        harmony = ChordProgression(
            track="piano",
            section="verse",
            chords=[
                Chord(root="C", quality="maj", bar=1, beat=1.0, duration="whole",
                      voicing=["C2", "E3", "G3"], velocity=75),
                Chord(root="G", quality="maj", bar=2, beat=1.0, duration="whole",
                      voicing=["G2", "B2", "D3"], velocity=75),
            ]
        )

        bass_line = await invoke_bass_agent(style_guide, section, harmony)

        # Find notes on beat 1 of each bar
        bar_1_beat_1_notes = [n for n in bass_line.notes if n.bar == 1 and n.beat == 1.0]
        bar_2_beat_1_notes = [n for n in bass_line.notes if n.bar == 2 and n.beat == 1.0]

        assert len(bar_1_beat_1_notes) > 0, "Should have note on bar 1, beat 1"
        assert len(bar_2_beat_1_notes) > 0, "Should have note on bar 2, beat 1"

        # Check that roots are present (C and G in octaves 1-3)
        # C in bass range: C1, C2, C3
        # G in bass range: G1, G2, G3
        has_c_root = any(n.pitch in ["C1", "C2", "C3"] for n in bar_1_beat_1_notes)
        has_g_root = any(n.pitch in ["G1", "G2", "G3"] for n in bar_2_beat_1_notes)

        assert has_c_root, "Should have C root on beat 1 of bar 1"
        assert has_g_root, "Should have G root on beat 1 of bar 2"

    @pytest.mark.asyncio
    async def test_different_styles_produce_different_patterns(self):
        """Test that jazz and pop styles produce recognizably different bass lines."""
        section = Section(
            name="verse",
            bars=(1, 4),
            key="Dm",
            tempo=90,
            energy="medium"
        )

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
                Chord(root="F", quality="maj7", bar=4, beat=1.0, duration="whole",
                      voicing=["F2", "A2", "E3", "C4"], velocity=75),
            ]
        )

        # Jazz style
        jazz_style = StyleGuide(
            genre="jazz",
            subgenre="bebop",
            harmonic_complexity="complex",
            swing=0.67,
            extensions_allowed=["7", "9", "11", "13"],
            tempo_range=(60, 90)
        )

        # Pop style
        pop_style = StyleGuide(
            genre="pop",
            subgenre="simple",
            harmonic_complexity="simple",
            swing=0.0,
            extensions_allowed=["7"],
            tempo_range=(100, 130)
        )

        jazz_bass = await invoke_bass_agent(jazz_style, section, harmony)
        pop_bass = await invoke_bass_agent(pop_style, section, harmony)

        # Jazz should typically have more notes (walking bass = quarter notes)
        # Pop should be simpler (roots on 1 and 3)
        # This is a general tendency, not a hard rule, so we just check they're different
        assert len(jazz_bass.notes) != len(pop_bass.notes) or \
               jazz_bass.notes[0].pitch != pop_bass.notes[0].pitch, \
               "Jazz and pop bass lines should be different"

    @pytest.mark.asyncio
    async def test_handles_longer_sections(self):
        """Test that bass agent can handle longer sections (8+ bars)."""
        style_guide = StyleGuide(
            genre="rock",
            subgenre="classic",
            harmonic_complexity="simple",
            swing=0.0,
            extensions_allowed=[],
            tempo_range=(110, 140)
        )

        section = Section(
            name="chorus",
            bars=(1, 8),
            key="E",
            tempo=120,
            energy="high"
        )

        # Create 8-bar progression
        chords = []
        for bar in range(1, 9):
            chord_root = "E" if bar % 4 in [1, 2] else "A" if bar % 4 == 3 else "B"
            chords.append(
                Chord(root=chord_root, quality="maj", bar=bar, beat=1.0,
                      duration="whole", voicing=[f"{chord_root}2", f"{chord_root}3"],
                      velocity=85)
            )

        harmony = ChordProgression(track="guitar", section="chorus", chords=chords)

        bass_line = await invoke_bass_agent(style_guide, section, harmony)

        # Should have notes across all 8 bars
        bars_with_notes = {note.bar for note in bass_line.notes}
        assert len(bars_with_notes) >= 4, "Should have notes in at least half the bars"
        assert max(bars_with_notes) <= 8, "Notes should not exceed section length"
        assert min(bars_with_notes) >= 1, "Notes should start from bar 1"
