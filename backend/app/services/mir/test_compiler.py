"""Unit tests for MIR compiler."""

import pytest
from app.services.mir.schema import Note, Chord, ChordProgression
from app.services.mir.compiler import (
    pitch_string_to_midi,
    beats_to_ticks,
    duration_to_ticks,
    compile_chord_to_notes,
    compile_progression_to_tool_calls,
    compile_note_to_midi,
)


class TestPitchStringToMidi:
    """Test pitch string to MIDI conversion."""

    def test_middle_c(self):
        """Test C4 = MIDI 60."""
        assert pitch_string_to_midi("C4") == 60

    def test_d4(self):
        """Test D4 = MIDI 62."""
        assert pitch_string_to_midi("D4") == 62

    def test_sharp_notes(self):
        """Test sharp notes."""
        assert pitch_string_to_midi("F#3") == 54
        assert pitch_string_to_midi("C#5") == 73

    def test_flat_notes(self):
        """Test flat notes."""
        assert pitch_string_to_midi("Bb2") == 46
        assert pitch_string_to_midi("Eb4") == 63

    def test_octave_range(self):
        """Test different octaves."""
        assert pitch_string_to_midi("C1") == 24
        assert pitch_string_to_midi("C2") == 36
        assert pitch_string_to_midi("C3") == 48
        assert pitch_string_to_midi("C5") == 72
        assert pitch_string_to_midi("C6") == 84

    def test_invalid_pitch_format(self):
        """Test invalid pitch format raises error."""
        with pytest.raises(ValueError):
            pitch_string_to_midi("X")

    def test_invalid_note(self):
        """Test invalid note name raises error."""
        with pytest.raises(ValueError):
            pitch_string_to_midi("H4")


class TestBeatsToTicks:
    """Test beats to ticks conversion."""

    def test_bar_1_beat_1(self):
        """Test bar 1, beat 1 = tick 0."""
        assert beats_to_ticks(1, 1.0) == 0

    def test_bar_1_beat_2(self):
        """Test bar 1, beat 2 = tick 480."""
        assert beats_to_ticks(1, 2.0) == 480

    def test_bar_2_beat_1(self):
        """Test bar 2, beat 1 = tick 1920 (4 beats * 480)."""
        assert beats_to_ticks(2, 1.0) == 1920

    def test_eighth_note_offset(self):
        """Test eighth note offset (beat 1.5)."""
        assert beats_to_ticks(1, 1.5) == 240

    def test_sixteenth_note_offset(self):
        """Test sixteenth note offset (beat 1.25)."""
        assert beats_to_ticks(1, 1.25) == 120

    def test_bar_3_beat_3(self):
        """Test bar 3, beat 3."""
        # Bar 3 starts at tick 3840 (2 bars * 1920)
        # Beat 3 adds 960 ticks (2 beats * 480)
        assert beats_to_ticks(3, 3.0) == 4800


class TestDurationToTicks:
    """Test duration to ticks conversion."""

    def test_whole_note(self):
        """Test whole note = 1920 ticks."""
        assert duration_to_ticks("whole") == 1920

    def test_half_note(self):
        """Test half note = 960 ticks."""
        assert duration_to_ticks("half") == 960

    def test_quarter_note(self):
        """Test quarter note = 480 ticks."""
        assert duration_to_ticks("quarter") == 480

    def test_eighth_note(self):
        """Test eighth note = 240 ticks."""
        assert duration_to_ticks("eighth") == 240

    def test_sixteenth_note(self):
        """Test sixteenth note = 120 ticks."""
        assert duration_to_ticks("sixteenth") == 120

    def test_unknown_duration_defaults_to_quarter(self):
        """Test unknown duration defaults to quarter note."""
        assert duration_to_ticks("unknown") == 480


class TestCompileChordToNotes:
    """Test chord to notes compilation."""

    def test_dm9_chord(self):
        """Test compiling Dm9 chord."""
        chord = Chord(
            root="D",
            quality="m9",
            bar=1,
            beat=1.0,
            duration="whole",
            voicing=["D2", "A2", "F3", "C4", "E4"],
            velocity=75
        )

        notes = compile_chord_to_notes(chord)

        assert len(notes) == 5
        assert notes[0]["pitch"] == pitch_string_to_midi("D2")
        assert notes[1]["pitch"] == pitch_string_to_midi("A2")
        assert notes[2]["pitch"] == pitch_string_to_midi("F3")
        assert notes[3]["pitch"] == pitch_string_to_midi("C4")
        assert notes[4]["pitch"] == pitch_string_to_midi("E4")

        # All notes should start at same time
        for note in notes:
            assert note["start"] == 0
            assert note["duration"] == 1920
            assert note["velocity"] == 75

    def test_chord_at_different_position(self):
        """Test chord at bar 2, beat 3."""
        chord = Chord(
            root="G",
            quality="7",
            bar=2,
            beat=3.0,
            duration="half",
            voicing=["G2", "D3", "F3", "B3"]
        )

        notes = compile_chord_to_notes(chord)

        expected_tick = beats_to_ticks(2, 3.0)  # Bar 2, beat 3
        for note in notes:
            assert note["start"] == expected_tick
            assert note["duration"] == 960  # half note


class TestCompileProgressionToToolCalls:
    """Test chord progression to tool calls compilation."""

    def test_simple_progression(self):
        """Test compiling a simple ii-V-I progression."""
        chords = [
            Chord(
                root="D",
                quality="m7",
                bar=1,
                beat=1.0,
                duration="whole",
                voicing=["D2", "A2", "C3", "F3"]
            ),
            Chord(
                root="G",
                quality="7",
                bar=2,
                beat=1.0,
                duration="whole",
                voicing=["G2", "D3", "F3", "B3"]
            ),
            Chord(
                root="C",
                quality="maj7",
                bar=3,
                beat=1.0,
                duration="whole",
                voicing=["C2", "G2", "B2", "E3"]
            )
        ]

        progression = ChordProgression(
            track="piano",
            section="verse_A",
            chords=chords
        )

        tool_calls = compile_progression_to_tool_calls(progression, track_id=1)

        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "addNotes"
        assert tool_calls[0]["args"]["trackId"] == 1

        notes = tool_calls[0]["args"]["notes"]
        assert len(notes) == 12  # 3 chords * 4 notes each

        # Verify notes are sorted by start time
        for i in range(len(notes) - 1):
            assert notes[i]["start"] <= notes[i + 1]["start"]

    def test_empty_progression(self):
        """Test compiling empty progression."""
        progression = ChordProgression(
            track="piano",
            section="intro",
            chords=[]
        )

        tool_calls = compile_progression_to_tool_calls(progression, track_id=2)

        assert len(tool_calls) == 1
        assert tool_calls[0]["args"]["notes"] == []


class TestCompileNoteToMidi:
    """Test single note to MIDI compilation."""

    def test_compile_note(self):
        """Test compiling a single note."""
        note = Note(
            pitch="D4",
            bar=1,
            beat=1.0,
            duration="quarter",
            velocity=80
        )

        midi_note = compile_note_to_midi(note)

        assert midi_note["pitch"] == 62
        assert midi_note["start"] == 0
        assert midi_note["duration"] == 480
        assert midi_note["velocity"] == 80

    def test_compile_note_with_offset(self):
        """Test compiling note with beat offset."""
        note = Note(
            pitch="F#3",
            bar=2,
            beat=2.5,
            duration="eighth",
            velocity=70
        )

        midi_note = compile_note_to_midi(note)

        expected_tick = beats_to_ticks(2, 2.5)
        assert midi_note["pitch"] == pitch_string_to_midi("F#3")
        assert midi_note["start"] == expected_tick
        assert midi_note["duration"] == 240  # eighth note
        assert midi_note["velocity"] == 70
