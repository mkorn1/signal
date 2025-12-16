"""MIR to MIDI Compiler.

Converts Musical Intermediate Representation objects to MIDI tool call format.
"""

from app.services.mir.schema import Chord, ChordProgression, Note, MelodyPhrase, DrumPattern, DrumHit, BassLine
from typing import List, Dict


# Music theory constants
PITCH_TO_MIDI = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "E#": 5, "Fb": 4,  # E# = F, Fb = E
    "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "B#": 0, "Cb": 11  # B# = C, Cb = B
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


def compile_melody_to_notes(phrase: MelodyPhrase, timebase: int = 480) -> List[Dict]:
    """Compile MelodyPhrase → addNotes format.

    Args:
        phrase: MelodyPhrase object to compile
        timebase: Ticks per quarter note (default 480)

    Returns:
        List of note dictionaries for addNotes tool
    """
    notes = []
    for note in phrase.notes:
        notes.append(compile_note_to_midi(note, timebase))

    # Sort by tick position
    notes.sort(key=lambda n: n["start"])

    return notes


def compile_drums_to_notes(pattern: DrumPattern, timebase: int = 480) -> List[Dict]:
    """Compile DrumPattern → addNotes format.

    Args:
        pattern: DrumPattern object to compile
        timebase: Ticks per quarter note (default 480)

    Returns:
        List of note dictionaries for addNotes tool
    """
    # Map drum instrument names to MIDI note numbers (General MIDI drum map)
    drum_map = {
        "kick": 36,
        "snare": 38,
        "hihat_closed": 42,
        "hihat_open": 46,
        "crash": 49,
        "ride": 51,
        "tom_low": 41,
        "tom_mid": 47,
        "tom_high": 50,
        "rim": 37,
    }

    notes = []
    for hit in pattern.hits:
        tick = beats_to_ticks(hit.bar, hit.beat, timebase)

        # Apply swing offset to off-beats
        if pattern.swing > 0 and (hit.beat % 1) >= 0.4 and (hit.beat % 1) <= 0.6:
            # Apply swing to notes on the "and" of the beat (x.5)
            swing_offset = int(pattern.swing * 240)  # Swing eighth notes
            tick += swing_offset

        # Map drum instrument name to MIDI note
        midi_pitch = drum_map.get(hit.instrument, 38)  # Default to snare if unknown

        notes.append({
            "pitch": midi_pitch,
            "start": tick,
            "duration": 120,  # Drums typically short (120 ticks = 1/4 of a quarter note)
            "velocity": hit.velocity
        })

    # Sort by tick position
    notes.sort(key=lambda n: n["start"])

    return notes


def compile_bass_to_notes(bass_line: BassLine, timebase: int = 480) -> List[Dict]:
    """Compile BassLine → addNotes format.

    Bass lines use the same Note format as melodies, so we can reuse
    the melody compilation logic.

    Args:
        bass_line: BassLine object to compile
        timebase: Ticks per quarter note (default 480)

    Returns:
        List of note dictionaries for addNotes tool
    """
    notes = []
    for note in bass_line.notes:
        notes.append(compile_note_to_midi(note, timebase))

    # Sort by tick position
    notes.sort(key=lambda n: n["start"])

    return notes
