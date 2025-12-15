"""Unit tests for Melody Agent."""

import pytest
from app.services.mir.schema import StyleGuide, Section, ChordProgression, Chord, MelodyPhrase
from app.services.agents.melody_agent import invoke_melody_agent
from app.services.mir.compiler import pitch_string_to_midi


@pytest.mark.asyncio
async def test_melody_agent_generates_valid_phrase():
    """Test that melody agent generates a valid MelodyPhrase."""
    style_guide = StyleGuide(
        genre="jazz",
        subgenre="ballad",
        harmonic_complexity="medium",
        swing=0.55,
        extensions_allowed=["7", "9"],
        tempo_range=(60, 85)
    )

    section = Section(
        name="verse_A",
        bars=(1, 8),
        key="Dm",
        tempo=72,
        energy="medium"
    )

    # Create a simple harmony
    harmony = ChordProgression(
        track="piano",
        section="verse_A",
        chords=[
            Chord(
                root="D",
                quality="m7",
                bar=1,
                beat=1.0,
                duration="whole",
                voicing=["D2", "A2", "C3", "F3"],
                function="tonic"
            ),
            Chord(
                root="G",
                quality="7",
                bar=2,
                beat=1.0,
                duration="whole",
                voicing=["G2", "D3", "F3", "B3"],
                function="dominant"
            )
        ]
    )

    # Invoke the melody agent
    melody = await invoke_melody_agent(style_guide, section, harmony, "flute")

    # Assertions
    assert isinstance(melody, MelodyPhrase)
    assert melody.track == "flute"
    assert melody.section == "verse_A"
    assert len(melody.notes) > 0

    # Check that all notes are in valid range (C4-G5)
    c4_midi = pitch_string_to_midi("C4")  # 60
    g5_midi = pitch_string_to_midi("G5")  # 79

    for note in melody.notes:
        midi_pitch = pitch_string_to_midi(note.pitch)
        assert c4_midi <= midi_pitch <= g5_midi, \
            f"Note {note.pitch} (MIDI {midi_pitch}) is out of range C4-G5 ({c4_midi}-{g5_midi})"


@pytest.mark.asyncio
async def test_melody_agent_matches_section_range():
    """Test that melody is generated within the section's bar range."""
    style_guide = StyleGuide(
        genre="pop",
        subgenre="ballad",
        harmonic_complexity="simple",
        swing=0.0,
        extensions_allowed=["sus4", "add9"],
        tempo_range=(70, 100)
    )

    section = Section(
        name="chorus_B",
        bars=(9, 16),
        key="C",
        tempo=85,
        energy="high"
    )

    harmony = ChordProgression(
        track="piano",
        section="chorus_B",
        chords=[
            Chord(
                root="C",
                quality="maj",
                bar=9,
                beat=1.0,
                duration="whole",
                voicing=["C2", "G2", "C3", "E3"]
            )
        ]
    )

    melody = await invoke_melody_agent(style_guide, section, harmony, "flute")

    # Check that all notes are within the section's bar range
    for note in melody.notes:
        assert section.bars[0] <= note.bar <= section.bars[1], \
            f"Note at bar {note.bar} is outside section range {section.bars}"


@pytest.mark.asyncio
async def test_melody_agent_uses_chord_tones():
    """Test that melody primarily uses chord tones on strong beats."""
    style_guide = StyleGuide(
        genre="jazz",
        subgenre="ballad",
        harmonic_complexity="medium",
        swing=0.55,
        extensions_allowed=["7", "9"],
        tempo_range=(60, 85)
    )

    section = Section(
        name="verse_A",
        bars=(1, 4),
        key="Dm",
        tempo=72,
        energy="medium"
    )

    # Simple Dm7 chord
    harmony = ChordProgression(
        track="piano",
        section="verse_A",
        chords=[
            Chord(
                root="D",
                quality="m7",
                bar=1,
                beat=1.0,
                duration="whole",
                voicing=["D2", "A2", "C3", "F3"],
                function="tonic"
            )
        ]
    )

    melody = await invoke_melody_agent(style_guide, section, harmony, "flute")

    # Dm7 chord tones in any octave: D, F, A, C
    dm7_chord_tones = {"D", "F", "A", "C"}

    # Check notes on strong beats (beat 1.0 or 3.0)
    strong_beat_notes = [n for n in melody.notes if n.beat in [1.0, 3.0]]

    if len(strong_beat_notes) > 0:
        # At least some strong beat notes should be chord tones
        chord_tone_count = sum(
            1 for note in strong_beat_notes
            if note.pitch[:-1] in dm7_chord_tones  # Remove octave number
        )

        # Relaxed check: at least 50% of strong beat notes should be chord tones
        assert chord_tone_count >= len(strong_beat_notes) * 0.5, \
            f"Only {chord_tone_count}/{len(strong_beat_notes)} strong beat notes are chord tones"
