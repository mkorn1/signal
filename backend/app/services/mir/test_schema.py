"""Unit tests for MIR schema."""

import pytest
from dataclasses import asdict
from app.services.mir.schema import Note, Chord, ChordProgression, Section, StyleGuide


class TestNote:
    """Test Note dataclass."""

    def test_note_creation(self):
        """Test creating a Note object."""
        note = Note(pitch="D4", bar=1, beat=1.0, duration="quarter", velocity=80)

        assert note.pitch == "D4"
        assert note.bar == 1
        assert note.beat == 1.0
        assert note.duration == "quarter"
        assert note.velocity == 80

    def test_note_default_velocity(self):
        """Test Note has default velocity of 80."""
        note = Note(pitch="F#3", bar=2, beat=2.5, duration="eighth")
        assert note.velocity == 80

    def test_note_serialization(self):
        """Test Note can be serialized to dict."""
        note = Note(pitch="Bb2", bar=3, beat=1.0, duration="half", velocity=90)
        note_dict = asdict(note)

        assert note_dict["pitch"] == "Bb2"
        assert note_dict["bar"] == 3
        assert note_dict["beat"] == 1.0
        assert note_dict["duration"] == "half"
        assert note_dict["velocity"] == 90


class TestChord:
    """Test Chord dataclass."""

    def test_chord_creation(self):
        """Test creating a Chord object."""
        chord = Chord(
            root="D",
            quality="m9",
            bar=1,
            beat=1.0,
            duration="whole",
            voicing=["D2", "A2", "F3", "C4", "E4"],
            function="tonic"
        )

        assert chord.root == "D"
        assert chord.quality == "m9"
        assert chord.bar == 1
        assert chord.beat == 1.0
        assert chord.duration == "whole"
        assert chord.voicing == ["D2", "A2", "F3", "C4", "E4"]
        assert chord.function == "tonic"

    def test_chord_optional_function(self):
        """Test Chord function is optional."""
        chord = Chord(
            root="G",
            quality="7",
            bar=2,
            beat=1.0,
            duration="whole",
            voicing=["G2", "D3", "F3", "B3"]
        )
        assert chord.function is None

    def test_chord_default_velocity(self):
        """Test Chord has default velocity of 75."""
        chord = Chord(
            root="C",
            quality="maj7",
            bar=1,
            beat=1.0,
            duration="whole",
            voicing=["C3", "E3", "G3", "B3"]
        )
        assert chord.velocity == 75


class TestChordProgression:
    """Test ChordProgression dataclass."""

    def test_progression_creation(self):
        """Test creating a ChordProgression."""
        chords = [
            Chord(root="D", quality="m7", bar=1, beat=1.0, duration="whole",
                  voicing=["D2", "A2", "C3", "F3"]),
            Chord(root="G", quality="7", bar=2, beat=1.0, duration="whole",
                  voicing=["G2", "D3", "F3", "B3"])
        ]

        progression = ChordProgression(
            track="piano",
            section="verse_A",
            chords=chords
        )

        assert progression.track == "piano"
        assert progression.section == "verse_A"
        assert len(progression.chords) == 2
        assert progression.chords[0].root == "D"
        assert progression.chords[1].root == "G"


class TestSection:
    """Test Section dataclass."""

    def test_section_creation(self):
        """Test creating a Section."""
        section = Section(
            name="intro",
            bars=(1, 8),
            key="Dm",
            tempo=72,
            energy="soft"
        )

        assert section.name == "intro"
        assert section.bars == (1, 8)
        assert section.key == "Dm"
        assert section.tempo == 72
        assert section.energy == "soft"

    def test_section_bars_tuple(self):
        """Test Section bars is a tuple."""
        section = Section(
            name="chorus_B",
            bars=(9, 24),
            key="F",
            tempo=120,
            energy="climax"
        )
        assert isinstance(section.bars, tuple)
        assert section.bars[0] == 9
        assert section.bars[1] == 24


class TestStyleGuide:
    """Test StyleGuide dataclass."""

    def test_style_guide_creation(self):
        """Test creating a StyleGuide."""
        style = StyleGuide(
            genre="jazz",
            subgenre="ballad",
            harmonic_complexity="complex",
            swing=0.55,
            extensions_allowed=["7", "9", "11", "13", "b9", "#11"],
            tempo_range=(60, 85)
        )

        assert style.genre == "jazz"
        assert style.subgenre == "ballad"
        assert style.harmonic_complexity == "complex"
        assert style.swing == 0.55
        assert "9" in style.extensions_allowed
        assert style.tempo_range == (60, 85)

    def test_style_guide_reference_artists_default(self):
        """Test StyleGuide reference_artists has default empty list."""
        style = StyleGuide(
            genre="rock",
            subgenre="classic",
            harmonic_complexity="simple",
            swing=0.0,
            extensions_allowed=["sus4"],
            tempo_range=(120, 140)
        )
        assert style.reference_artists == []

    def test_style_guide_with_reference_artists(self):
        """Test StyleGuide with reference artists."""
        style = StyleGuide(
            genre="jazz",
            subgenre="ballad",
            harmonic_complexity="complex",
            swing=0.55,
            extensions_allowed=["7", "9", "11", "13"],
            tempo_range=(60, 85),
            reference_artists=["Bill Evans", "Chet Baker"]
        )
        assert len(style.reference_artists) == 2
        assert "Bill Evans" in style.reference_artists
