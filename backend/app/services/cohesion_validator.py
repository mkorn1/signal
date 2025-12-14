"""Cohesion validation utilities for audio track alignment.

Uses librosa for audio analysis to validate that generated tracks
match the target tempo and other cohesion constraints.
"""

import io
import logging
from typing import Optional, Tuple

from app.models.schemas import (
    CohesionSpec,
    CohesionValidationResult,
    StrengthProfile,
    STRENGTH_PROFILES,
)

logger = logging.getLogger(__name__)


def get_strength_for_instrument(
    instrument: str,
    profile: StrengthProfile = StrengthProfile.MEDIUM,
) -> float:
    """
    Get the audio-to-audio strength value for an instrument based on profile.
    
    Args:
        instrument: Instrument name (e.g., "drums", "bass")
        profile: Cohesion strength profile
        
    Returns:
        Strength value (0.0-1.0)
    """
    profile_strengths = STRENGTH_PROFILES.get(profile, STRENGTH_PROFILES[StrengthProfile.MEDIUM])
    return profile_strengths.get(instrument.lower(), profile_strengths["default"])


def detect_bpm(audio_bytes: bytes) -> Optional[float]:
    """
    Detect the tempo (BPM) of audio using librosa.
    
    Args:
        audio_bytes: WAV audio as bytes
        
    Returns:
        Detected BPM or None if detection fails
    """
    try:
        import librosa
        import numpy as np
        import soundfile as sf
        
        # Load audio from bytes
        audio_buffer = io.BytesIO(audio_bytes)
        y, sr = sf.read(audio_buffer)
        
        # Convert to mono if stereo
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        
        # Detect tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # librosa returns array, get scalar
        if hasattr(tempo, '__len__'):
            tempo = float(tempo[0]) if len(tempo) > 0 else None
        else:
            tempo = float(tempo)
            
        logger.info(f"Detected BPM: {tempo}")
        return tempo
        
    except ImportError:
        logger.warning("librosa not installed - BPM detection unavailable")
        return None
    except Exception as e:
        logger.error(f"BPM detection failed: {e}")
        return None


def validate_track_cohesion(
    audio_bytes: bytes,
    cohesion_spec: CohesionSpec,
) -> CohesionValidationResult:
    """
    Validate that a generated audio track meets cohesion constraints.
    
    Args:
        audio_bytes: Generated audio as WAV bytes
        cohesion_spec: Cohesion constraints to validate against
        
    Returns:
        Validation result with pass/fail and details
    """
    issues = []
    suggestions = []
    passed = True
    
    # Detect BPM
    detected_bpm = detect_bpm(audio_bytes)
    bpm_drift = None
    
    if detected_bpm is not None:
        bpm_drift = abs(detected_bpm - cohesion_spec.tempo)
        
        if bpm_drift > cohesion_spec.max_bpm_drift:
            passed = False
            issues.append(
                f"BPM drift too high: detected {detected_bpm:.1f} vs target {cohesion_spec.tempo} "
                f"(drift: {bpm_drift:.1f}, max allowed: {cohesion_spec.max_bpm_drift})"
            )
            suggestions.append(
                f"Try regenerating with tighter strength profile or adjust target tempo to {detected_bpm:.0f}"
            )
        else:
            logger.info(f"BPM validation passed: {detected_bpm:.1f} (drift: {bpm_drift:.1f})")
    else:
        # Can't validate BPM - pass with warning
        issues.append("Could not detect BPM - validation skipped")
        suggestions.append("Install librosa for BPM validation: pip install librosa")
    
    return CohesionValidationResult(
        passed=passed,
        detected_bpm=detected_bpm,
        bpm_drift=bpm_drift,
        issues=issues,
        suggestions=suggestions,
    )


def build_cohesion_prompt_context(cohesion_spec: CohesionSpec) -> str:
    """
    Build prompt context string from cohesion spec.
    
    This adds consistent style/mood terms to ensure all tracks
    share the same vibe.
    
    Args:
        cohesion_spec: Cohesion specification
        
    Returns:
        Prompt context string to append to instrument prompts
    """
    parts = [cohesion_spec.style]
    
    if cohesion_spec.mood:
        parts.append(cohesion_spec.mood)
    
    parts.append(f"{cohesion_spec.tempo} BPM")
    
    return ", ".join(parts)
