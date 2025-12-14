"""Stem separation agent - generates full track then separates into instrument stems.

Uses:
1. MusicGen to generate a complete mixed music track
2. Demucs to separate the track into individual instrument stems

This approach produces a more cohesive-sounding final mix since all parts
were generated together, then professionally separated.
"""

import logging
from typing import Dict
from app.services.audio_renderer import generate_full_track, separate_stems_demucs

logger = logging.getLogger(__name__)


async def generate_with_stem_separation(
    prompt: str, tempo: int = 120, key: str = "Am"
) -> Dict[str, bytes]:
    """
    Generate a complete track, then separate into instrument stems using Demucs.
    
    Strategy:
    1. Use MusicGen to generate a full, mixed music track
    2. Use Demucs to separate into drums, bass, other, and melody stems
    
    This produces stems that are guaranteed to work together musically
    since they came from the same original generation.
    
    Args:
        prompt: Style/genre description (e.g., "upbeat indie rock")
        tempo: BPM for the track
        key: Musical key
        
    Returns:
        Dict mapping stem names to audio bytes (WAV)
        Stems: drums, bass, other (guitars/keys), melody
    """
    logger.info(f"Generating full track: {prompt}")
    
    # Step 1: Generate complete mixed track
    full_audio = await generate_full_track(
        style=prompt,
        duration=30,  # 30 seconds
        tempo=tempo,
        key=key,
    )
    logger.info(f"Generated full track: {len(full_audio)} bytes")
    
    # Step 2: Separate into stems using Demucs
    logger.info("Separating into stems with Demucs...")
    stems = await separate_stems_demucs(full_audio)
    logger.info(f"Separated into {len(stems)} stems: {list(stems.keys())}")
    
    return stems
