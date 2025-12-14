"""Per-instrument track generation agent - generates audio for each instrument separately.

Uses Stability AI's Stable Audio 2 to generate audio directly for each instrument.

Strategy:
1. Generate the first instrument (typically melody) using text-to-audio
2. Use audio-to-audio for subsequent instruments, with the first track as reference
3. This ensures all stems share the same harmonic content and rhythm

The audio-to-audio feature guides new generations using the reference audio,
so subsequent instruments will naturally match the key, tempo, and feel of the first.

Cohesion System:
- CohesionSpec defines constraints for track alignment (style, tempo, duration, strength)
- Post-generation validation checks BPM drift
- Strength profiles (tight/medium/loose) control how closely tracks match
"""

import asyncio
import base64
import logging
from typing import Callable, Coroutine, Dict, List, Optional, Union

from app.services.audio_renderer import generate_audio_for_instrument, generate_audio_to_audio
from app.models.schemas import CohesionSpec, StrengthProfile, CohesionValidationResult
from app.services.cohesion_validator import (
    get_strength_for_instrument,
    validate_track_cohesion,
    build_cohesion_prompt_context,
)

logger = logging.getLogger(__name__)

# Default instruments to generate based on common music styles
# Melody/lead is always first for sequential generation
STYLE_INSTRUMENTS = {
    "rock": ["melody", "drums", "bass", "guitar"],
    "pop": ["melody", "drums", "bass", "keys"],
    "electronic": ["melody", "drums", "bass", "synth"],
    "jazz": ["melody", "drums", "bass", "piano"],
    "acoustic": ["melody", "guitar", "bass", "piano"],
    "orchestral": ["melody", "strings", "bass", "pad"],
    "default": ["melody", "drums", "bass", "keys"],
}

# Instrument generation order priority (melody always first)
INSTRUMENT_PRIORITY = {
    "melody": 0,
    "lead": 0,
    "drums": 1,
    "bass": 2,
    "guitar": 3,
    "keys": 4,
    "piano": 4,
    "synth": 5,
    "strings": 6,
    "pad": 7,
}


def _detect_instruments_from_prompt(prompt: str) -> List[str]:
    """Detect which instruments to generate based on the prompt."""
    prompt_lower = prompt.lower()

    # Check for explicit instrument mentions
    explicit_instruments = []
    instrument_keywords = {
        "drums": ["drums", "drum", "percussion", "beat"],
        "bass": ["bass", "bassline"],
        "guitar": ["guitar", "guitars"],
        "piano": ["piano", "keys", "keyboard"],
        "synth": ["synth", "synthesizer", "electronic"],
        "strings": ["strings", "orchestra", "violin", "cello"],
        "melody": ["melody", "lead", "vocal"],
        "pad": ["pad", "ambient", "atmosphere"],
    }

    for instrument, keywords in instrument_keywords.items():
        if any(kw in prompt_lower for kw in keywords):
            explicit_instruments.append(instrument)

    if explicit_instruments:
        # Ensure melody is first
        if "melody" not in explicit_instruments and "lead" not in explicit_instruments:
            explicit_instruments.insert(0, "melody")
        # Sort by priority
        explicit_instruments.sort(key=lambda x: INSTRUMENT_PRIORITY.get(x, 10))
        return explicit_instruments[:5]  # Max 5 instruments

    # Detect style and use default instruments
    for style, instruments in STYLE_INSTRUMENTS.items():
        if style in prompt_lower:
            return instruments

    return STYLE_INSTRUMENTS["default"]


def _get_instrument_context_prompt(
    instrument: str,
    style: str,
    tempo: int,
    key: str,  # kept for API compatibility but not used in prompt (doesn't affect Stable Audio)
    previous_instruments: List[str],
) -> str:
    """
    Build a high-quality, context-aware prompt for each instrument.
    Uses descriptive adjectives and production quality terms for better output.
    """
    # Production quality suffix added to all prompts
    quality_suffix = "professional studio quality, well-mixed, high fidelity audio"
    
    # Instrument-specific prompts with rich descriptive adjectives
    base_prompts = {
        "melody": (
            f"{style}, {tempo} BPM, "
            f"expressive lead melody, memorable melodic hook, soaring main theme, "
            f"clear and prominent in the mix, emotional and captivating, "
            f"melody instrument only"
        ),
        "lead": (
            f"{style}, {tempo} BPM, "
            f"expressive lead instrument, dynamic solo melody, articulate phrasing, "
            f"front and center in the mix, emotionally engaging, "
            f"lead instrument only"
        ),
        "drums": (
            f"{style}, {tempo} BPM, "
            f"punchy drum kit, tight snare with crack, deep kick drum with punch, "
            f"crisp hi-hats, solid groove, professional drum sound, "
            f"well-balanced drum mix, drums and percussion only, no melodic instruments"
        ),
        "bass": (
            f"{style}, {tempo} BPM, "
            f"deep and warm bass, punchy low end with definition, "
            f"rhythmic bass groove, clean sub frequencies, "
            f"sitting perfectly in the mix, bass only, no drums no melody"
        ),
        "guitar": (
            f"{style}, {tempo} BPM, "
            f"rich guitar tone, clear and articulate playing, "
            f"warm and full-bodied sound, professional guitar recording, "
            f"guitar only, no drums no bass"
        ),
        "keys": (
            f"{style}, {tempo} BPM, "
            f"warm keyboard sound, lush chords, smooth and full texture, "
            f"professional keyboard recording, atmospheric harmonic support, "
            f"keys only, no drums no bass no melody"
        ),
        "piano": (
            f"{style}, {tempo} BPM, "
            f"rich acoustic piano, warm and resonant tone, expressive dynamics, "
            f"professional piano recording, clear and detailed, "
            f"piano only, no other instruments"
        ),
        "synth": (
            f"{style}, {tempo} BPM, "
            f"lush synthesizer, rich electronic textures, warm analog-style sound, "
            f"atmospheric and immersive, professional synth programming, "
            f"synth only, no drums no bass"
        ),
        "strings": (
            f"{style}, {tempo} BPM, "
            f"lush orchestral strings, warm and emotive, cinematic string arrangement, "
            f"rich harmonic layers, professional orchestral recording, "
            f"strings only, no percussion no bass"
        ),
        "pad": (
            f"{style}, {tempo} BPM, "
            f"atmospheric pad, lush ambient texture, warm and enveloping sound, "
            f"spacious and ethereal, professional ambient production, "
            f"pad only, no rhythmic elements"
        ),
    }

    base = base_prompts.get(
        instrument.lower(),
        f"{style}, {tempo} BPM, {instrument} track, professional quality recording, "
        f"clear and well-defined, {instrument} only, no other instruments"
    )

    # Add quality suffix
    base += f", {quality_suffix}"

    return base


ProgressCallback = Callable[[str, str], Union[None, Coroutine]]


async def _call_progress(callback: Optional[ProgressCallback], instrument: str, status: str):
    """Call progress callback, handling both sync and async callbacks."""
    if callback is None:
        return
    result = callback(instrument, status)
    if asyncio.iscoroutine(result):
        await result


async def generate_per_instrument(
    prompt: str,
    tempo: int = 120,
    key: str = "Am",
    progress_callback: Optional[ProgressCallback] = None,
    use_audio_to_audio: bool = True,
    cohesion_spec: Optional[CohesionSpec] = None,
    validate_cohesion: bool = False,
    max_retries: int = 2,
) -> Dict[str, bytes]:
    """
    Generate audio for each instrument sequentially using Stable Audio.

    Strategy: Generate the first instrument (melody) with text-to-audio, then use
    audio-to-audio with the first track as reference for subsequent instruments.
    This ensures all stems are harmonically and rhythmically coherent.

    Args:
        prompt: Style/genre description (e.g., "upbeat indie rock")
        tempo: BPM for the tracks
        key: Musical key (kept for API compat, not used in prompts)
        progress_callback: Optional callback for progress updates (instrument_name, status)
        use_audio_to_audio: If True, use audio-to-audio for subsequent instruments (default: True)
        cohesion_spec: Optional cohesion constraints for track alignment
        validate_cohesion: If True, validate BPM after generation (requires librosa)
        max_retries: Max regeneration attempts if validation fails

    Returns:
        Dict mapping instrument names to audio bytes (WAV)
    """
    # Determine which instruments to generate (melody first)
    instruments = _detect_instruments_from_prompt(prompt)
    logger.info(f"Generating {len(instruments)} instruments sequentially: {instruments}")

    # Extract cohesion settings
    strength_profile = cohesion_spec.strength_profile if cohesion_spec else None
    duration = cohesion_spec.duration if cohesion_spec else 20
    target_tempo = cohesion_spec.tempo if cohesion_spec else tempo
    
    # Build style context from cohesion spec if provided
    style_context = prompt
    if cohesion_spec:
        style_context = build_cohesion_prompt_context(cohesion_spec)
        logger.info(f"Using cohesion style context: {style_context}")

    audio_files: Dict[str, bytes] = {}
    validation_results: Dict[str, CohesionValidationResult] = {}
    
    # Use provided reference audio or None (first generated becomes reference)
    reference_audio: Optional[bytes] = None
    if cohesion_spec and cohesion_spec.reference_audio:
        reference_audio = base64.b64decode(cohesion_spec.reference_audio)
        logger.info("Using provided reference audio from cohesion spec")

    # Generate each instrument sequentially, starting with melody
    for i, instrument in enumerate(instruments):
        await _call_progress(progress_callback, instrument, "generating")

        # Build context-aware prompt with cohesion context
        instrument_prompt = _get_instrument_context_prompt(
            instrument=instrument,
            style=style_context,
            tempo=target_tempo,
            key=key,
            previous_instruments=list(audio_files.keys()),
        )

        logger.info(f"Generating {instrument} track (#{i+1}/{len(instruments)})")

        # Retry loop for validation failures
        for attempt in range(max_retries + 1):
            try:
                if (i == 0 and reference_audio is None) or not use_audio_to_audio:
                    # First instrument (no reference): use text-to-audio
                    logger.info(f"Using text-to-audio for {instrument}: {instrument_prompt[:80]}...")
                    audio = await generate_audio_for_instrument(
                        instrument=instrument,
                        style=instrument_prompt,
                        duration=duration,
                        tempo=target_tempo,
                        key=key,
                        use_raw_style=True,
                    )
                    # Store as reference for subsequent tracks (if no external reference)
                    if reference_audio is None:
                        reference_audio = audio
                else:
                    # Subsequent instruments: use audio-to-audio with reference
                    strength = _get_audio_to_audio_strength(instrument, strength_profile)
                    logger.info(f"Using audio-to-audio for {instrument} (strength={strength}): {instrument_prompt[:80]}...")
                    audio = await generate_audio_to_audio(
                        reference_audio=reference_audio,
                        prompt=instrument_prompt,
                        duration=duration,
                        strength=strength,
                    )

                # Validate cohesion if enabled
                if validate_cohesion and cohesion_spec:
                    await _call_progress(progress_callback, instrument, "validating")
                    validation = validate_track_cohesion(audio, cohesion_spec)
                    validation_results[instrument] = validation
                    
                    if not validation.passed:
                        logger.warning(f"Cohesion validation failed for {instrument}: {validation.issues}")
                        if attempt < max_retries:
                            logger.info(f"Retrying {instrument} (attempt {attempt + 2}/{max_retries + 1})")
                            await _call_progress(progress_callback, instrument, f"retry ({attempt + 2})")
                            continue
                        else:
                            logger.warning(f"Max retries reached for {instrument}, keeping last result")

                audio_files[instrument] = audio
                await _call_progress(progress_callback, instrument, "complete")
                logger.info(f"Generated {instrument} track: {len(audio)} bytes")
                break  # Success, exit retry loop

            except Exception as e:
                logger.error(f"Failed to generate {instrument}: {e}")
                await _call_progress(progress_callback, instrument, f"error: {e}")
                
                # If audio-to-audio fails, try falling back to text-to-audio
                if reference_audio is not None and use_audio_to_audio:
                    logger.info(f"Falling back to text-to-audio for {instrument}")
                    try:
                        audio = await generate_audio_for_instrument(
                            instrument=instrument,
                            style=instrument_prompt,
                            duration=duration,
                            tempo=target_tempo,
                            key=key,
                            use_raw_style=True,
                        )
                        audio_files[instrument] = audio
                        await _call_progress(progress_callback, instrument, "complete (fallback)")
                        logger.info(f"Fallback succeeded for {instrument}: {len(audio)} bytes")
                        break
                    except Exception as e2:
                        logger.error(f"Fallback also failed for {instrument}: {e2}")
                break  # Exit retry loop on error

    return audio_files


async def generate_single_instrument(
    instrument: str,
    prompt: str,
    cohesion_spec: CohesionSpec,
    progress_callback: Optional[ProgressCallback] = None,
    validate_cohesion: bool = True,
    max_retries: int = 2,
) -> tuple[bytes, Optional[CohesionValidationResult]]:
    """
    Generate a single instrument track with cohesion constraints.
    
    Use this for adding a new instrument to an existing song.
    
    Args:
        instrument: Instrument to generate (e.g., "bass", "drums")
        prompt: Additional style context
        cohesion_spec: Cohesion constraints (must include reference_audio)
        progress_callback: Optional progress callback
        validate_cohesion: Whether to validate BPM alignment
        max_retries: Max regeneration attempts
        
    Returns:
        Tuple of (audio_bytes, validation_result)
    """
    if not cohesion_spec.reference_audio:
        raise ValueError("cohesion_spec.reference_audio is required for single instrument generation")
    
    reference_audio = base64.b64decode(cohesion_spec.reference_audio)
    style_context = build_cohesion_prompt_context(cohesion_spec)
    
    # Merge user prompt with cohesion context
    if prompt:
        full_style = f"{prompt}, {style_context}"
    else:
        full_style = style_context
    
    instrument_prompt = _get_instrument_context_prompt(
        instrument=instrument,
        style=full_style,
        tempo=cohesion_spec.tempo,
        key="",  # Not used
        previous_instruments=[],
    )
    
    strength = _get_audio_to_audio_strength(instrument, cohesion_spec.strength_profile)
    validation_result = None
    
    for attempt in range(max_retries + 1):
        await _call_progress(progress_callback, instrument, "generating")
        
        audio = await generate_audio_to_audio(
            reference_audio=reference_audio,
            prompt=instrument_prompt,
            duration=cohesion_spec.duration,
            strength=strength,
        )
        
        if validate_cohesion:
            await _call_progress(progress_callback, instrument, "validating")
            validation_result = validate_track_cohesion(audio, cohesion_spec)
            
            if not validation_result.passed and attempt < max_retries:
                logger.warning(f"Validation failed, retrying: {validation_result.issues}")
                continue
        
        await _call_progress(progress_callback, instrument, "complete")
        return audio, validation_result
    
    # Return last attempt even if validation failed
    return audio, validation_result


def _get_audio_to_audio_strength(
    instrument: str,
    strength_profile: Optional[StrengthProfile] = None,
) -> float:
    """
    Get the appropriate audio-to-audio strength for each instrument type.
    
    Lower strength = stays closer to reference (better for harmonic instruments)
    Higher strength = more transformation (better for rhythmic instruments)
    
    Args:
        instrument: Instrument name
        strength_profile: Optional profile override (tight/medium/loose)
        
    Returns:
        Strength value (0.0-1.0)
    """
    if strength_profile is not None:
        return get_strength_for_instrument(instrument, strength_profile)
    
    # Legacy default behavior (equivalent to MEDIUM profile)
    strength_map = {
        "drums": 0.65,      # Higher - focus on rhythm/groove, not pitch
        "bass": 0.40,       # Lower - needs to match harmonic content
        "guitar": 0.45,     # Medium-low - harmonic instrument
        "keys": 0.45,       # Medium-low - harmonic instrument
        "piano": 0.45,      # Medium-low - harmonic instrument
        "synth": 0.50,      # Medium - can vary more
        "strings": 0.45,    # Medium-low - harmonic instrument
        "pad": 0.50,        # Medium - atmospheric, can vary
        "melody": 0.40,     # Low - melodic content should match
        "lead": 0.40,       # Low - melodic content should match
    }
    return strength_map.get(instrument.lower(), 0.50)
