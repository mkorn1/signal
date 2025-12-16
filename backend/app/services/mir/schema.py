"""Musical Intermediate Representation (MIR) Schema.

This module defines the data structures for representing music in a human-readable,
music-theory-aware format before compilation to MIDI.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class Note:
    """Single note in MIR - uses music notation, not MIDI numbers."""
    pitch: str  # "D4", "F#3", "Bb2"
    bar: int
    beat: float  # 1.0 = downbeat, 1.5 = eighth note offset
    duration: str  # "whole", "half", "quarter", "eighth", "sixteenth"
    velocity: int = 80


@dataclass
class Chord:
    """Chord in MIR - harmonic thinking, not individual notes yet."""
    root: str  # "D", "F#", "Bb"
    quality: str  # "m9", "maj7", "7b9", "sus4", etc.
    bar: int
    beat: float
    duration: str
    voicing: List[str]  # ["D2", "A2", "F3", "C4", "E4"] - explicit pitches
    function: Optional[str] = None  # "tonic", "dominant", "subdominant", "passing"
    velocity: int = 75


@dataclass
class ChordProgression:
    """Container for harmony in a section."""
    track: str  # "piano", "guitar"
    section: str  # "verse_A", "chorus_B"
    chords: List[Chord]


@dataclass
class Section:
    """Song structure element."""
    name: str  # "intro", "verse_A", "chorus_B"
    bars: tuple[int, int]  # (start_bar, end_bar)
    key: str  # "Dm", "F", "A"
    tempo: int
    energy: str  # "soft", "building", "climax", "resolve"


@dataclass
class StyleGuide:
    """Style contract - all agents must follow this."""
    genre: str  # "jazz"
    subgenre: str  # "ballad"
    harmonic_complexity: str  # "complex" (9ths, 11ths), "medium" (7ths), "simple" (triads)
    swing: float  # 0.0 (straight), 0.55 (ballad), 0.67 (bebop)
    extensions_allowed: List[str]  # ["9", "11", "13", "b9", "#11"]
    tempo_range: tuple[int, int]
    reference_artists: List[str] = field(default_factory=list)
