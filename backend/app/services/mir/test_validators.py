"""Unit tests for MIR validators."""

import pytest
from app.services.mir.schema import Chord, ChordProgression, Note, MelodyPhrase, StyleGuide
from app.services.mir.validators import (
    validate_voice_leading,
    validate_melody_range,
    validate_style_consistency,
    validate_all
)


class TestVoiceLeadingValidator:
    """Tests for validate_voice_leading function."""

    def test_no_issues_with_good_voice_leading(self):
        """Test that good voice leading passes without issues."""
        # Create a simple ii-V-I progression with smooth voice leading
        progression = ChordProgression(
            track="piano",
            section="test",
            chords=[
                Chord(
                    root="D",
                    quality="m7",
                    bar=1,
                    beat=1.0,
                    duration="whole",
                    voicing=["D2", "A2", "F3", "C4"],  # Dm7
                    velocity=75
                ),
                Chord(
                    root="G",
                    quality="7",
                    bar=2,
                    beat=1.0,
                    duration="whole",
                    voicing=["G2", "B2", "F3", "D4"],  # G7 - smooth voice leading
                    velocity=75
                ),
                Chord(
                    root="C",
                    quality="maj7",
                    bar=3,
                    beat=1.0,
                    duration="whole",
                    voicing=["C2", "B2", "E3", "C4"],  # Cmaj7 - smooth voice leading
                    velocity=75
                )
            ]
        )

        issues = validate_voice_leading(progression)
        assert len(issues) == 0, "Good voice leading should produce no issues"

    def test_detects_parallel_fifths(self):
        """Test that parallel fifths are detected."""
        # Create progression with parallel fifths
        progression = ChordProgression(
            track="piano",
            section="test",
            chords=[
                Chord(
                    root="C",
                    quality="maj",
                    bar=1,
                    beat=1.0,
                    duration="whole",
                    voicing=["C3", "G3", "C4"],  # C major
                    velocity=75
                ),
                Chord(
                    root="D",
                    quality="maj",
                    bar=2,
                    beat=1.0,
                    duration="whole",
                    voicing=["D3", "A3", "D4"],  # D major - parallel motion up
                    velocity=75
                )
            ]
        )

        issues = validate_voice_leading(progression)
        # Should detect parallel fifths
        parallel_fifth_issues = [i for i in issues if i["type"] == "parallel_fifth"]
        assert len(parallel_fifth_issues) > 0, "Should detect parallel fifths"
        assert all(i["severity"] == "error" for i in parallel_fifth_issues)

    def test_detects_large_jumps(self):
        """Test that large jumps (>7 semitones) are detected."""
        progression = ChordProgression(
            track="piano",
            section="test",
            chords=[
                Chord(
                    root="C",
                    quality="maj",
                    bar=1,
                    beat=1.0,
                    duration="whole",
                    voicing=["C2", "E3", "G3"],
                    velocity=75
                ),
                Chord(
                    root="C",
                    quality="maj",
                    bar=2,
                    beat=1.0,
                    duration="whole",
                    voicing=["C2", "E3", "G4"],  # Top voice jumps an octave (12 semitones)
                    velocity=75
                )
            ]
        )

        issues = validate_voice_leading(progression)
        large_jump_issues = [i for i in issues if i["type"] == "large_jump"]
        assert len(large_jump_issues) > 0, "Should detect large jumps"
        assert all(i["severity"] == "warning" for i in large_jump_issues)

    def test_handles_invalid_pitch(self):
        """Test that invalid pitches are caught gracefully."""
        progression = ChordProgression(
            track="piano",
            section="test",
            chords=[
                Chord(
                    root="C",
                    quality="maj",
                    bar=1,
                    beat=1.0,
                    duration="whole",
                    voicing=["C2", "INVALID", "G3"],
                    velocity=75
                ),
                Chord(
                    root="G",
                    quality="maj",
                    bar=2,
                    beat=1.0,
                    duration="whole",
                    voicing=["G2", "B2", "D3"],
                    velocity=75
                )
            ]
        )

        issues = validate_voice_leading(progression)
        invalid_pitch_issues = [i for i in issues if i["type"] == "invalid_pitch"]
        assert len(invalid_pitch_issues) > 0, "Should catch invalid pitch"


class TestMelodyRangeValidator:
    """Tests for validate_melody_range function."""

    def test_melody_within_range(self):
        """Test that melody within range passes."""
        phrase = MelodyPhrase(
            track="flute",
            section="verse",
            notes=[
                Note(pitch="C4", bar=1, beat=1.0, duration="quarter", velocity=80),
                Note(pitch="E4", bar=1, beat=2.0, duration="quarter", velocity=80),
                Note(pitch="G4", bar=1, beat=3.0, duration="quarter", velocity=80),
                Note(pitch="C5", bar=1, beat=4.0, duration="quarter", velocity=80),
            ]
        )

        issues = validate_melody_range(phrase, max_pitch="G5", min_pitch="C4")
        assert len(issues) == 0, "Melody within range should pass"

    def test_detects_notes_above_max(self):
        """Test that notes above max range are detected."""
        phrase = MelodyPhrase(
            track="flute",
            section="verse",
            notes=[
                Note(pitch="G5", bar=1, beat=1.0, duration="quarter", velocity=80),
                Note(pitch="A5", bar=1, beat=2.0, duration="quarter", velocity=80),  # Above G5
            ]
        )

        issues = validate_melody_range(phrase, max_pitch="G5", min_pitch="C4")
        range_issues = [i for i in issues if i["type"] == "range_violation"]
        assert len(range_issues) == 1, "Should detect note above max"
        assert range_issues[0]["severity"] == "error"
        assert "A5" in range_issues[0]["message"]

    def test_detects_notes_below_min(self):
        """Test that notes below min range are detected."""
        phrase = MelodyPhrase(
            track="flute",
            section="verse",
            notes=[
                Note(pitch="C4", bar=1, beat=1.0, duration="quarter", velocity=80),
                Note(pitch="B3", bar=1, beat=2.0, duration="quarter", velocity=80),  # Below C4
            ]
        )

        issues = validate_melody_range(phrase, max_pitch="G5", min_pitch="C4")
        range_issues = [i for i in issues if i["type"] == "range_violation"]
        assert len(range_issues) == 1, "Should detect note below min"
        assert range_issues[0]["severity"] == "error"
        assert "B3" in range_issues[0]["message"]

    def test_handles_invalid_pitch(self):
        """Test that invalid pitches are caught gracefully."""
        phrase = MelodyPhrase(
            track="flute",
            section="verse",
            notes=[
                Note(pitch="INVALID", bar=1, beat=1.0, duration="quarter", velocity=80),
            ]
        )

        issues = validate_melody_range(phrase)
        invalid_pitch_issues = [i for i in issues if i["type"] == "invalid_pitch"]
        assert len(invalid_pitch_issues) > 0, "Should catch invalid pitch"


class TestStyleConsistencyValidator:
    """Tests for validate_style_consistency function."""

    def test_allows_permitted_extensions(self):
        """Test that permitted extensions pass."""
        style_guide = StyleGuide(
            genre="jazz",
            subgenre="ballad",
            harmonic_complexity="complex",
            swing=0.55,
            extensions_allowed=["7", "9", "11", "13"],
            tempo_range=(60, 90)
        )

        progression = ChordProgression(
            track="piano",
            section="verse",
            chords=[
                Chord(root="D", quality="m9", bar=1, beat=1.0, duration="whole",
                      voicing=["D2", "A2", "F3", "C4", "E4"], velocity=75),
                Chord(root="G", quality="13", bar=2, beat=1.0, duration="whole",
                      voicing=["G2", "B2", "F3", "E4"], velocity=75),
            ]
        )

        issues = validate_style_consistency(progression, style_guide)
        assert len(issues) == 0, "Permitted extensions should pass"

    def test_detects_forbidden_extensions(self):
        """Test that forbidden extensions are flagged."""
        style_guide = StyleGuide(
            genre="pop",
            subgenre="simple",
            harmonic_complexity="simple",
            swing=0.0,
            extensions_allowed=["7"],  # Only 7ths allowed
            tempo_range=(100, 130)
        )

        progression = ChordProgression(
            track="piano",
            section="verse",
            chords=[
                Chord(root="C", quality="maj", bar=1, beat=1.0, duration="whole",
                      voicing=["C2", "E3", "G3"], velocity=75),
                Chord(root="D", quality="m9", bar=2, beat=1.0, duration="whole",  # 9th not allowed
                      voicing=["D2", "A2", "F3", "E4"], velocity=75),
            ]
        )

        issues = validate_style_consistency(progression, style_guide)
        style_issues = [i for i in issues if i["type"] == "style_violation"]
        assert len(style_issues) > 0, "Should detect forbidden extension"
        assert all(i["severity"] == "warning" for i in style_issues)
        assert any("9" in i["message"] for i in style_issues)

    def test_handles_altered_extensions(self):
        """Test that altered extensions are allowed if base extension is allowed."""
        style_guide = StyleGuide(
            genre="jazz",
            subgenre="bebop",
            harmonic_complexity="complex",
            swing=0.67,
            extensions_allowed=["7", "9", "13"],  # b9 allowed because 9 is allowed
            tempo_range=(180, 240)
        )

        progression = ChordProgression(
            track="piano",
            section="verse",
            chords=[
                Chord(root="G", quality="7b9", bar=1, beat=1.0, duration="whole",
                      voicing=["G2", "B2", "F3", "Ab3"], velocity=75),
            ]
        )

        issues = validate_style_consistency(progression, style_guide)
        style_issues = [i for i in issues if i["type"] == "style_violation"]
        # b9 should be allowed because base extension "9" is in the allowed list
        # This is reasonable for jazz where altered extensions are common
        assert len(style_issues) == 0, "Altered extensions should be allowed if base extension is allowed"

    def test_detects_truly_forbidden_extensions(self):
        """Test that extensions with no base in allowed list are flagged."""
        style_guide = StyleGuide(
            genre="pop",
            subgenre="simple",
            harmonic_complexity="simple",
            swing=0.0,
            extensions_allowed=["7"],  # Only 7ths, no 11ths at all
            tempo_range=(100, 130)
        )

        progression = ChordProgression(
            track="piano",
            section="verse",
            chords=[
                Chord(root="F", quality="maj7#11", bar=1, beat=1.0, duration="whole",
                      voicing=["F2", "A2", "E3", "B3"], velocity=75),  # #11 not allowed
            ]
        )

        issues = validate_style_consistency(progression, style_guide)
        style_issues = [i for i in issues if i["type"] == "style_violation"]
        # #11 should be flagged because "11" is not in allowed list
        assert len(style_issues) > 0, "Should detect extension with no base in allowed list"


class TestValidateAll:
    """Tests for validate_all convenience function."""

    def test_validates_all_components(self):
        """Test that validate_all runs all validators."""
        style_guide = StyleGuide(
            genre="jazz",
            subgenre="ballad",
            harmonic_complexity="complex",
            swing=0.55,
            extensions_allowed=["7", "9"],
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
                      voicing=["D3", "A3", "D4"], velocity=75),  # Parallel fifths
            ]
        )

        # Create melody with range violation
        melody = MelodyPhrase(
            track="flute",
            section="verse",
            notes=[
                Note(pitch="A5", bar=1, beat=1.0, duration="quarter", velocity=80),  # Above G5
            ]
        )

        issues = validate_all(harmony=harmony, melody=melody, style_guide=style_guide)

        # Should have issues from both validators
        assert len(issues) > 0
        issue_types = {i["type"] for i in issues}
        assert "parallel_fifth" in issue_types or "large_jump" in issue_types
        assert "range_violation" in issue_types

    def test_handles_none_arguments(self):
        """Test that validate_all handles None arguments gracefully."""
        issues = validate_all(harmony=None, melody=None, style_guide=None)
        assert len(issues) == 0, "Should handle None arguments"
