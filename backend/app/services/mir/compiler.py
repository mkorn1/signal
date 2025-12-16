"""MIR to MIDI Compiler.

Converts Musical Intermediate Representation objects to MIDI tool call format.
"""

from app.services.mir.schema import Chord, ChordProgression, Note
from typing import List, Dict


# Music theory constants
PITCH_TO_MIDI = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11
}

DURATION_TO_TICKS = {
    "whole": 1920, "half": 960, "quarter": 480,
    "eighth": 240, "sixteenth": 120, "thirtysecond": 60
}


def pitch_string_to_midi(pitch: str) -> int:
    """Convert 'D4' → 62, 'F#3' → 54.

    Args:
        pitch: Pitch string like "D4", "F#3", "Bb2"

    Returns:
        MIDI note number (0-127)

    Raises:
        ValueError: If pitch format is invalid
    """
    if len(pitch) < 2:
        raise ValueError(f"Invalid pitch format: {pitch}")

    # Extract note and octave
    # Handle both single character (C4) and double character (C#4, Bb4)
    if len(pitch) == 2:
        note = pitch[0]
        octave = int(pitch[1])
    else:
        note = pitch[:-1]
        octave = int(pitch[-1])

    if note not in PITCH_TO_MIDI:
        raise ValueError(f"Invalid note: {note}")

    # MIDI note calculation: (octave + 1) * 12 + pitch_class
    # C4 = 60, so octave 4 is at base 60
    midi_note = PITCH_TO_MIDI[note] + (octave + 1) * 12

    if midi_note < 0 or midi_note > 127:
        raise ValueError(f"MIDI note {midi_note} out of range (0-127) for pitch {pitch}")

    return midi_note


def beats_to_ticks(bar: int, beat: float, timebase: int = 480) -> int:
    """Convert musical time (bar 2, beat 1.5) → tick position.

    Assumes 4/4 time signature.

    Args:
        bar: Bar number (1-indexed)
        beat: Beat number (1.0 = downbeat, 1.5 = eighth note after downbeat)
        timebase: Ticks per quarter note (default 480)

    Returns:
        Tick position
    """
    # Bar 1 starts at tick 0
    ticks_per_bar = timebase * 4  # 4 beats per bar in 4/4 time
    tick = (bar - 1) * ticks_per_bar + int((beat - 1) * timebase)
    return max(0, tick)


def duration_to_ticks(duration: str, timebase: int = 480) -> int:
    """Convert 'quarter' → 480 ticks.

    Args:
        duration: Duration string like "quarter", "eighth", "whole"
        timebase: Ticks per quarter note (default 480)

    Returns:
        Duration in ticks
    """
    return DURATION_TO_TICKS.get(duration, 480)


def compile_chord_to_notes(chord: Chord, timebase: int = 480) -> List[Dict]:
    """Compile a Chord MIR object to MIDI note format.

    Returns list of notes for addNotes tool:
    [{"pitch": 62, "start": 0, "duration": 1920, "velocity": 75}, ...]

    Args:
        chord: Chord object to compile
        timebase: Ticks per quarter note (default 480)

    Returns:
        List of note dictionaries
    """
    tick = beats_to_ticks(chord.bar, chord.beat, timebase)
    duration_ticks = duration_to_ticks(chord.duration, timebase)

    notes = []
    for pitch_str in chord.voicing:
        midi_pitch = pitch_string_to_midi(pitch_str)
        notes.append({
            "pitch": midi_pitch,
            "start": tick,
            "duration": duration_ticks,
            "velocity": chord.velocity
        })

    return notes


def compile_progression_to_tool_calls(
    progression: ChordProgression,
    track_id: int,
    timebase: int = 480
) -> List[Dict]:
    """Compile ChordProgression → addNotes tool calls.

    Args:
        progression: ChordProgression object
        track_id: Target track ID
        timebase: Ticks per quarter note (default 480)

    Returns:
        List of tool call dictionaries:
        [{"name": "addNotes", "args": {"trackId": 1, "notes": [...]}}]
    """
    all_notes = []
    for chord in progression.chords:
        all_notes.extend(compile_chord_to_notes(chord, timebase))

    # Sort by tick position
    all_notes.sort(key=lambda n: n["start"])

    return [{
        "name": "addNotes",
        "args": {
            "trackId": track_id,
            "notes": all_notes
        }
    }]


def compile_note_to_midi(note: Note, timebase: int = 480) -> Dict:
    """Compile a single Note MIR object to MIDI format.

    Args:
        note: Note object to compile
        timebase: Ticks per quarter note (default 480)

    Returns:
        Note dictionary for addNotes tool
    """
    tick = beats_to_ticks(note.bar, note.beat, timebase)
    duration_ticks = duration_to_ticks(note.duration, timebase)
    midi_pitch = pitch_string_to_midi(note.pitch)

    return {
        "pitch": midi_pitch,
        "start": tick,
        "duration": duration_ticks,
        "velocity": note.velocity
    }
