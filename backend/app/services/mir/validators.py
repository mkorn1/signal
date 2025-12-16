"""MIR Validators - Rule-based quality checks for musical content.

This module provides validators that check musical correctness:
- Voice leading (parallel fifths/octaves, excessive jumps)
- Melody range (singable range validation)
- Style consistency (extension usage matches style guide)
"""

from app.services.mir.schema import Chord, ChordProgression, MelodyPhrase, StyleGuide
from app.services.mir.compiler import pitch_string_to_midi
from typing import List, Dict


def validate_voice_leading(progression: ChordProgression) -> List[Dict]:
    """Check for voice leading errors (parallel fifths, excessive jumps).

    Args:
        progression: ChordProgression to validate

    Returns:
        List of issue dictionaries with type, severity, location, message
    """
    errors = []

    for i in range(len(progression.chords) - 1):
        curr_chord = progression.chords[i]
        next_chord = progression.chords[i + 1]

        # Convert voicings to MIDI numbers
        try:
            curr_pitches = [pitch_string_to_midi(p) for p in curr_chord.voicing]
            next_pitches = [pitch_string_to_midi(p) for p in next_chord.voicing]
        except (ValueError, IndexError) as e:
            errors.append({
                "type": "invalid_pitch",
                "severity": "error",
                "location": f"bar {curr_chord.bar}",
                "message": f"Invalid pitch in voicing: {str(e)}",
                "agent": "harmony"
            })
            continue

        # Check for parallel fifths/octaves
        # Only check if both chords have at least 2 voices
        if len(curr_pitches) >= 2 and len(next_pitches) >= 2:
            for voice_idx in range(1, min(len(curr_pitches), len(next_pitches))):
                # Calculate intervals between this voice and the bass (voice 0)
                interval_curr = (curr_pitches[voice_idx] - curr_pitches[0]) % 12
                interval_next = (next_pitches[voice_idx] - next_pitches[0]) % 12

                # Check if both intervals are perfect fifths (7) or octaves (0)
                if interval_curr in [0, 7] and interval_curr == interval_next:
                    # Check if voices are moving in same direction (parallel motion)
                    voice_motion = next_pitches[voice_idx] - curr_pitches[voice_idx]
                    bass_motion = next_pitches[0] - curr_pitches[0]

                    # If both voices move in same direction, it's parallel motion
                    if (voice_motion > 0 and bass_motion > 0) or (voice_motion < 0 and bass_motion < 0):
                        interval_name = "octave" if interval_curr == 0 else "fifth"
                        errors.append({
                            "type": "parallel_fifth",
                            "severity": "error",
                            "location": f"bar {curr_chord.bar} to {next_chord.bar}",
                            "message": f"Parallel {interval_name}s between bass and voice {voice_idx}",
                            "agent": "harmony",
                            "suggestion": "Use contrary motion in inner voices"
                        })

        # Check for excessive jumps (>7 semitones / perfect fifth in any voice)
        for voice_idx in range(min(len(curr_pitches), len(next_pitches))):
            jump = abs(next_pitches[voice_idx] - curr_pitches[voice_idx])
            if jump > 7:  # More than a perfect fifth
                errors.append({
                    "type": "large_jump",
                    "severity": "warning",
                    "location": f"bar {curr_chord.bar} to {next_chord.bar}",
                    "message": f"Voice {voice_idx} jumps {jump} semitones",
                    "agent": "harmony",
                    "suggestion": "Consider stepwise motion for smoother voice leading"
                })

    return errors


def validate_melody_range(
    phrase: MelodyPhrase,
    max_pitch: str = "G5",
    min_pitch: str = "C4"
) -> List[Dict]:
    """Check melody stays in singable range.

    Args:
        phrase: MelodyPhrase to validate
        max_pitch: Maximum pitch (default "G5")
        min_pitch: Minimum pitch (default "C4")

    Returns:
        List of issue dictionaries
    """
    errors = []

    try:
        max_midi = pitch_string_to_midi(max_pitch)
        min_midi = pitch_string_to_midi(min_pitch)
    except ValueError as e:
        errors.append({
            "type": "invalid_range",
            "severity": "error",
            "location": "range specification",
            "message": f"Invalid range specification: {str(e)}",
            "agent": "melody"
        })
        return errors

    for note in phrase.notes:
        try:
            midi_pitch = pitch_string_to_midi(note.pitch)
        except ValueError as e:
            errors.append({
                "type": "invalid_pitch",
                "severity": "error",
                "location": f"bar {note.bar}",
                "message": f"Invalid pitch {note.pitch}: {str(e)}",
                "agent": "melody"
            })
            continue

        if midi_pitch > max_midi:
            errors.append({
                "type": "range_violation",
                "severity": "error",
                "location": f"bar {note.bar}, beat {note.beat}",
                "message": f"Note {note.pitch} (MIDI {midi_pitch}) exceeds max range {max_pitch} (MIDI {max_midi})",
                "agent": "melody",
                "suggestion": f"Keep melody below {max_pitch}"
            })
        elif midi_pitch < min_midi:
            errors.append({
                "type": "range_violation",
                "severity": "error",
                "location": f"bar {note.bar}, beat {note.beat}",
                "message": f"Note {note.pitch} (MIDI {midi_pitch}) below min range {min_pitch} (MIDI {min_midi})",
                "agent": "melody",
                "suggestion": f"Keep melody above {min_pitch}"
            })

    return errors


def validate_style_consistency(
    harmony: ChordProgression,
    style_guide: StyleGuide
) -> List[Dict]:
    """Check if harmony uses only allowed extensions.

    Args:
        harmony: ChordProgression to validate
        style_guide: StyleGuide to check against

    Returns:
        List of issue dictionaries
    """
    errors = []

    for chord in harmony.chords:
        # Parse quality to extract extensions (e.g., "m9" → ["9"])
        extensions = []
        quality = chord.quality

        # Check for various extensions in the chord quality string
        extension_patterns = {
            "9": ["9", "add9"],
            "11": ["11", "add11"],
            "13": ["13", "add13"],
            "b9": ["b9", "♭9"],
            "#9": ["#9", "♯9"],
            "#11": ["#11", "♯11"],
            "b13": ["b13", "♭13"],
        }

        for ext, patterns in extension_patterns.items():
            for pattern in patterns:
                if pattern in quality:
                    extensions.append(ext)
                    break

        # Check if any found extensions are not in the allowed list
        for ext in extensions:
            # Normalize the extension (remove sharps/flats for base comparison)
            base_ext = ext.replace('b', '').replace('#', '')

            # Check if this extension or its base is allowed
            is_allowed = (
                ext in style_guide.extensions_allowed or
                base_ext in style_guide.extensions_allowed
            )

            if not is_allowed:
                errors.append({
                    "type": "style_violation",
                    "severity": "warning",
                    "location": f"bar {chord.bar}",
                    "message": f"Extension '{ext}' in chord {chord.root}{chord.quality} not typically used in {style_guide.genre} {style_guide.subgenre}",
                    "agent": "harmony",
                    "suggestion": f"Consider using simpler extensions from: {', '.join(style_guide.extensions_allowed)}"
                })

    return errors


def validate_all(
    harmony: ChordProgression = None,
    melody: MelodyPhrase = None,
    style_guide: StyleGuide = None
) -> List[Dict]:
    """Run all validators and return combined issues.

    Args:
        harmony: Optional ChordProgression to validate
        melody: Optional MelodyPhrase to validate
        style_guide: Optional StyleGuide for style consistency checks

    Returns:
        Combined list of all issues found
    """
    all_issues = []

    if harmony:
        all_issues.extend(validate_voice_leading(harmony))

        if style_guide:
            all_issues.extend(validate_style_consistency(harmony, style_guide))

    if melody:
        all_issues.extend(validate_melody_range(melody))

    return all_issues
