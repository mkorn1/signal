"""Unit tests for MIR compiler extensions (melody and rhythm)."""

import pytest
from app.services.mir.schema import Note, MelodyPhrase, DrumPattern, DrumHit
from app.services.mir.compiler import (
    compile_melody_to_notes,
    compile_drums_to_notes,
    pitch_string_to_midi,
    beats_to_ticks,
)


class TestCompileMelodyToNotes:
    """Test melody phrase compilation."""

    def test_simple_melody(self):
        """Test compiling a simple melodic phrase."""
        notes = [
            Note(pitch="D4", bar=1, beat=1.0, duration="quarter", velocity=80),
            Note(pitch="F4", bar=1, beat=2.0, duration="eighth", velocity=75),
            Note(pitch="E4", bar=1, beat=2.5, duration="eighth", velocity=70),
        ]

        phrase = MelodyPhrase(
            track="flute",
            section="verse_A",
            notes=notes,
            motif_id="motif_A"
        )

        midi_notes = compile_melody_to_notes(phrase)

        assert len(midi_notes) == 3
        assert midi_notes[0]["pitch"] == pitch_string_to_midi("D4")
        assert midi_notes[0]["start"] == 0
        assert midi_notes[0]["duration"] == 480  # quarter
        assert midi_notes[0]["velocity"] == 80

        assert midi_notes[1]["pitch"] == pitch_string_to_midi("F4")
        assert midi_notes[1]["start"] == 480  # beat 2
        assert midi_notes[1]["duration"] == 240  # eighth
        assert midi_notes[1]["velocity"] == 75

        assert midi_notes[2]["pitch"] == pitch_string_to_midi("E4")
        assert midi_notes[2]["start"] == 720  # beat 2.5
        assert midi_notes[2]["duration"] == 240  # eighth
        assert midi_notes[2]["velocity"] == 70

    def test_melody_range_c4_to_g5(self):
        """Test melody in the C4-G5 range."""
        notes = [
            Note(pitch="C4", bar=1, beat=1.0, duration="quarter", velocity=80),
            Note(pitch="G5", bar=1, beat=2.0, duration="quarter", velocity=80),
        ]

        phrase = MelodyPhrase(
            track="flute",
            section="verse_A",
            notes=notes
        )

        midi_notes = compile_melody_to_notes(phrase)

        assert len(midi_notes) == 2
        assert midi_notes[0]["pitch"] == pitch_string_to_midi("C4")  # 60
        assert midi_notes[1]["pitch"] == pitch_string_to_midi("G5")  # 79

    def test_melody_sorted_by_time(self):
        """Test notes are sorted by start time."""
        notes = [
            Note(pitch="E4", bar=2, beat=1.0, duration="quarter", velocity=80),
            Note(pitch="D4", bar=1, beat=1.0, duration="quarter", velocity=80),
            Note(pitch="F4", bar=1, beat=3.0, duration="quarter", velocity=80),
        ]

        phrase = MelodyPhrase(
            track="flute",
            section="verse_A",
            notes=notes
        )

        midi_notes = compile_melody_to_notes(phrase)

        # Should be sorted: bar 1 beat 1, bar 1 beat 3, bar 2 beat 1
        assert len(midi_notes) == 3
        assert midi_notes[0]["start"] == 0  # bar 1, beat 1
        assert midi_notes[1]["start"] == 960  # bar 1, beat 3
        assert midi_notes[2]["start"] == 1920  # bar 2, beat 1

    def test_empty_melody(self):
        """Test compiling empty melody phrase."""
        phrase = MelodyPhrase(
            track="flute",
            section="intro",
            notes=[]
        )

        midi_notes = compile_melody_to_notes(phrase)

        assert len(midi_notes) == 0


class TestCompileDrumsToNotes:
    """Test drum pattern compilation."""

    def test_basic_drum_pattern(self):
        """Test compiling a basic drum pattern (kick, snare, hi-hat)."""
        hits = [
            DrumHit(instrument="kick", bar=1, beat=1.0, velocity=100),
            DrumHit(instrument="snare", bar=1, beat=2.0, velocity=90),
            DrumHit(instrument="hihat_closed", bar=1, beat=1.0, velocity=70),
            DrumHit(instrument="hihat_closed", bar=1, beat=1.5, velocity=60),
        ]

        pattern = DrumPattern(
            track="drums",
            section="verse_A",
            hits=hits,
            swing=0.0,  # Straight feel
            variation_every_n_bars=4
        )

        midi_notes = compile_drums_to_notes(pattern)

        assert len(midi_notes) == 4

        # Check drum MIDI note mappings
        kick_notes = [n for n in midi_notes if n["pitch"] == 36]
        snare_notes = [n for n in midi_notes if n["pitch"] == 38]
        hihat_notes = [n for n in midi_notes if n["pitch"] == 42]

        assert len(kick_notes) == 1
        assert len(snare_notes) == 1
        assert len(hihat_notes) == 2

        # Verify kick on beat 1
        assert kick_notes[0]["start"] == 0
        assert kick_notes[0]["velocity"] == 100

        # Verify snare on beat 2
        assert snare_notes[0]["start"] == 480
        assert snare_notes[0]["velocity"] == 90

        # Verify hi-hats
        assert hihat_notes[0]["velocity"] == 70
        assert hihat_notes[1]["velocity"] == 60

    def test_drum_mapping(self):
        """Test all drum instrument mappings."""
        hits = [
            DrumHit(instrument="kick", bar=1, beat=1.0, velocity=100),
            DrumHit(instrument="snare", bar=1, beat=2.0, velocity=90),
            DrumHit(instrument="hihat_closed", bar=1, beat=3.0, velocity=70),
            DrumHit(instrument="hihat_open", bar=1, beat=4.0, velocity=75),
            DrumHit(instrument="crash", bar=2, beat=1.0, velocity=95),
            DrumHit(instrument="ride", bar=2, beat=2.0, velocity=80),
        ]

        pattern = DrumPattern(
            track="drums",
            section="verse_A",
            hits=hits,
            swing=0.0
        )

        midi_notes = compile_drums_to_notes(pattern)

        # Check MIDI note numbers match General MIDI drum map
        drum_pitches = {n["pitch"] for n in midi_notes}
        assert 36 in drum_pitches  # kick
        assert 38 in drum_pitches  # snare
        assert 42 in drum_pitches  # hihat_closed
        assert 46 in drum_pitches  # hihat_open
        assert 49 in drum_pitches  # crash
        assert 51 in drum_pitches  # ride

    def test_swing_offset(self):
        """Test swing feel applies offset to off-beats."""
        hits = [
            DrumHit(instrument="hihat_closed", bar=1, beat=1.0, velocity=70),
            DrumHit(instrument="hihat_closed", bar=1, beat=1.5, velocity=60),
        ]

        pattern = DrumPattern(
            track="drums",
            section="verse_A",
            hits=hits,
            swing=0.55  # Jazz swing
        )

        midi_notes = compile_drums_to_notes(pattern)

        # Beat 1.0 should be at tick 0 (no swing)
        on_beat = [n for n in midi_notes if n["velocity"] == 70][0]
        assert on_beat["start"] == 0

        # Beat 1.5 should have swing offset applied
        off_beat = [n for n in midi_notes if n["velocity"] == 60][0]
        expected_tick = 240  # Base tick for beat 1.5
        swing_offset = int(0.55 * 240)  # Swing amount
        expected_with_swing = expected_tick + swing_offset

        # Allow small rounding differences
        assert abs(off_beat["start"] - expected_with_swing) <= 1

    def test_drum_duration(self):
        """Test drums have short duration (120 ticks)."""
        hits = [
            DrumHit(instrument="kick", bar=1, beat=1.0, velocity=100),
        ]

        pattern = DrumPattern(
            track="drums",
            section="verse_A",
            hits=hits,
            swing=0.0
        )

        midi_notes = compile_drums_to_notes(pattern)

        assert len(midi_notes) == 1
        assert midi_notes[0]["duration"] == 120  # Short duration for drums

    def test_drums_sorted_by_time(self):
        """Test drum hits are sorted by start time."""
        hits = [
            DrumHit(instrument="crash", bar=2, beat=1.0, velocity=95),
            DrumHit(instrument="kick", bar=1, beat=1.0, velocity=100),
            DrumHit(instrument="snare", bar=1, beat=3.0, velocity=90),
        ]

        pattern = DrumPattern(
            track="drums",
            section="verse_A",
            hits=hits,
            swing=0.0
        )

        midi_notes = compile_drums_to_notes(pattern)

        # Should be sorted by start time
        assert len(midi_notes) == 3
        assert midi_notes[0]["start"] < midi_notes[1]["start"] < midi_notes[2]["start"]

    def test_empty_drum_pattern(self):
        """Test compiling empty drum pattern."""
        pattern = DrumPattern(
            track="drums",
            section="intro",
            hits=[],
            swing=0.0
        )

        midi_notes = compile_drums_to_notes(pattern)

        assert len(midi_notes) == 0

    def test_unknown_drum_instrument_defaults_to_snare(self):
        """Test unknown drum instrument defaults to snare (38)."""
        hits = [
            DrumHit(instrument="unknown_drum", bar=1, beat=1.0, velocity=80),
        ]

        pattern = DrumPattern(
            track="drums",
            section="verse_A",
            hits=hits,
            swing=0.0
        )

        midi_notes = compile_drums_to_notes(pattern)

        assert len(midi_notes) == 1
        assert midi_notes[0]["pitch"] == 38  # Defaults to snare
