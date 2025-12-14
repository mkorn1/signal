"""Audio generation and stem separation utilities.

Uses:
- Stability AI's Stable Audio 2 for text-to-music generation
- Demucs for high-quality stem separation (local or via Replicate)

Setup:
1. Get a Stability API key from https://platform.stability.ai
2. Set STABILITY_API_KEY in your .env file
3. (Optional) Set REPLICATE_API_TOKEN for stem separation if no local GPU
"""

import os
import io
import tempfile
import logging
from typing import Dict, Optional
import httpx
from app.config import get_settings

logger = logging.getLogger(__name__)

STABILITY_TEXT_TO_AUDIO_URL = "https://api.stability.ai/v2beta/audio/stable-audio-2/text-to-audio"
STABILITY_AUDIO_TO_AUDIO_URL = "https://api.stability.ai/v2beta/audio/stable-audio-2/audio-to-audio"


def _get_stability_key() -> Optional[str]:
    """Get Stability API key from settings or environment."""
    settings = get_settings()
    return settings.stability_api_key or os.getenv("STABILITY_API_KEY")


def _get_replicate_token() -> Optional[str]:
    """Get Replicate API token from settings or environment."""
    settings = get_settings()
    return settings.replicate_api_token or os.getenv("REPLICATE_API_TOKEN")


async def generate_audio_for_instrument(
    instrument: str,
    style: str,
    duration: int = 10,
    tempo: int = 120,
    key: str = "Am",
    use_raw_style: bool = False,
) -> bytes:
    """
    Generate audio for a specific instrument using Stability AI's Stable Audio 2.
    
    Args:
        instrument: Name of instrument (e.g., "drums", "bass", "guitar")
        style: Style/genre description or full prompt if use_raw_style=True
        duration: Duration in seconds (5-180)
        tempo: BPM for the track
        key: Musical key
        use_raw_style: If True, use style as the full prompt without modification
        
    Returns:
        Audio bytes (WAV format)
    """
    api_key = _get_stability_key()
    if not api_key:
        raise ValueError("STABILITY_API_KEY not set - cannot generate audio")
    
    # Use raw style as prompt or build instrument-specific prompt
    if use_raw_style:
        prompt = style
    else:
        prompt = _build_instrument_prompt(instrument, style, tempo, key)
    logger.info(f"Generating {instrument} with prompt: {prompt}")
    
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                STABILITY_TEXT_TO_AUDIO_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Accept": "audio/*",
                },
                files={
                    "prompt": (None, prompt),
                    "duration": (None, str(min(max(duration, 5), 180))),
                    "output_format": (None, "wav"),
                },
            )
            
            if response.status_code == 200:
                logger.info(f"Generated {instrument} track: {len(response.content)} bytes")
                return response.content
            else:
                error_msg = response.text
                logger.error(f"Stable Audio generation failed: {response.status_code} - {error_msg}")
                raise RuntimeError(f"Stable Audio generation failed: {response.status_code} - {error_msg}")
                
    except httpx.TimeoutException:
        logger.error("Stable Audio generation timed out")
        raise RuntimeError("Audio generation timed out - try a shorter duration")
    except Exception as e:
        logger.error(f"Error generating audio with Stable Audio: {e}")
        raise


async def generate_audio_to_audio(
    reference_audio: bytes,
    prompt: str,
    duration: int = 20,
    strength: float = 0.5,
) -> bytes:
    """
    Generate audio using Stable Audio 2's audio-to-audio feature.
    
    Uses the reference audio as a guide for the new generation, allowing
    subsequent instruments to harmonically match the reference.
    
    Args:
        reference_audio: Reference audio bytes (WAV format) to guide generation
        prompt: Text prompt describing what to generate
        duration: Duration in seconds (5-180)
        strength: How much to transform from reference (0=identical, 1=ignore reference)
                  Lower values stay closer to the reference's harmonic/rhythmic content.
                  Recommended: 0.3-0.6 for matching harmony while changing instrument.
        
    Returns:
        Audio bytes (WAV format)
    """
    api_key = _get_stability_key()
    if not api_key:
        raise ValueError("STABILITY_API_KEY not set - cannot generate audio")
    
    logger.info(f"Audio-to-audio generation with strength={strength}: {prompt[:100]}...")
    
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                STABILITY_AUDIO_TO_AUDIO_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Accept": "audio/*",
                },
                files={
                    "audio": ("reference.wav", reference_audio, "audio/wav"),
                    "prompt": (None, prompt),
                    "duration": (None, str(min(max(duration, 5), 180))),
                    "strength": (None, str(strength)),
                    "output_format": (None, "wav"),
                },
            )
            
            if response.status_code == 200:
                logger.info(f"Audio-to-audio generated: {len(response.content)} bytes")
                return response.content
            else:
                error_msg = response.text
                logger.error(f"Stable Audio audio-to-audio failed: {response.status_code} - {error_msg}")
                raise RuntimeError(f"Stable Audio audio-to-audio failed: {response.status_code} - {error_msg}")
                
    except httpx.TimeoutException:
        logger.error("Stable Audio audio-to-audio timed out")
        raise RuntimeError("Audio generation timed out - try a shorter duration")
    except Exception as e:
        logger.error(f"Error with audio-to-audio: {e}")
        raise


def _build_instrument_prompt(instrument: str, style: str, tempo: int, key: str) -> str:
    """Build a high-quality, detailed prompt for instrument-specific generation.
    
    Uses descriptive adjectives and production quality terms for better Stable Audio output.
    Note: key parameter kept for API compatibility but not used (Stable Audio doesn't understand keys well).
    """
    # Quality suffix for all instruments
    quality = "professional studio quality, high fidelity, well-mixed"
    
    instrument_prompts = {
        "drums": (
            f"{style} drum track, {tempo} BPM, "
            f"punchy kick drum with deep low end, tight snappy snare, crisp hi-hats, "
            f"solid consistent groove, professional drum recording, "
            f"drums and percussion only, no melodic instruments, no bass, {quality}"
        ),
        "bass": (
            f"{style} bass, {tempo} BPM, "
            f"deep warm bass tone, punchy low end with clear definition, "
            f"rhythmic groove, clean sub frequencies, "
            f"bass only, no drums, no melody, {quality}"
        ),
        "guitar": (
            f"{style} guitar, {tempo} BPM, "
            f"rich warm guitar tone, clear articulate playing, "
            f"full-bodied sound, professional guitar recording, "
            f"guitar only, no drums, no bass, {quality}"
        ),
        "keys": (
            f"{style} keyboard, {tempo} BPM, "
            f"warm lush keyboard sound, smooth full texture, "
            f"atmospheric harmonic support, professional keyboard recording, "
            f"keys only, no drums, no bass, {quality}"
        ),
        "piano": (
            f"{style} piano, {tempo} BPM, "
            f"rich acoustic piano, warm resonant tone, expressive dynamics, "
            f"clear detailed recording, professional piano sound, "
            f"piano only, no other instruments, {quality}"
        ),
        "synth": (
            f"{style} synthesizer, {tempo} BPM, "
            f"lush warm synth, rich electronic textures, analog-style warmth, "
            f"atmospheric and immersive, professional synth production, "
            f"synth only, no drums, no bass, {quality}"
        ),
        "melody": (
            f"{style} lead melody, {tempo} BPM, "
            f"expressive memorable melodic line, soaring emotional theme, "
            f"clear and prominent, captivating hook, "
            f"melody only, no drums, no bass, {quality}"
        ),
        "lead": (
            f"{style} lead instrument, {tempo} BPM, "
            f"expressive dynamic solo melody, articulate phrasing, "
            f"emotionally engaging, front and center, "
            f"lead only, no drums, no bass, {quality}"
        ),
        "strings": (
            f"{style} orchestral strings, {tempo} BPM, "
            f"lush warm string ensemble, emotive cinematic arrangement, "
            f"rich harmonic layers, professional orchestral recording, "
            f"strings only, no drums, no percussion, {quality}"
        ),
        "pad": (
            f"{style} ambient pad, {tempo} BPM, "
            f"lush atmospheric texture, warm enveloping sound, "
            f"spacious ethereal ambience, professional ambient production, "
            f"pad only, no rhythmic elements, no drums, {quality}"
        ),
    }
    
    return instrument_prompts.get(
        instrument.lower(),
        f"{style} {instrument}, {tempo} BPM, professional quality recording, "
        f"clear well-defined sound, {instrument} only, no other instruments, {quality}"
    )


async def generate_full_track(
    style: str,
    duration: int = 30,
    tempo: int = 120,
    key: str = "Am",
) -> bytes:
    """
    Generate a complete music track using Stability AI's Stable Audio 2.
    
    Args:
        style: Style/genre description
        duration: Duration in seconds (5-180)
        tempo: BPM
        key: Musical key
        
    Returns:
        Audio bytes (WAV format)
    """
    api_key = _get_stability_key()
    if not api_key:
        raise ValueError("STABILITY_API_KEY not set - cannot generate audio")
    
    prompt = f"{style} music track in {key}, {tempo} BPM, full arrangement with drums, bass, and melody, high quality production"
    logger.info(f"Generating full track with prompt: {prompt}")
    
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                STABILITY_TEXT_TO_AUDIO_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Accept": "audio/*",
                },
                files={
                    "prompt": (None, prompt),
                    "duration": (None, str(min(max(duration, 5), 180))),
                    "output_format": (None, "wav"),
                },
            )
            
            if response.status_code == 200:
                logger.info(f"Generated full track: {len(response.content)} bytes")
                return response.content
            else:
                error_msg = response.text
                logger.error(f"Stable Audio generation failed: {response.status_code} - {error_msg}")
                raise RuntimeError(f"Stable Audio generation failed: {response.status_code} - {error_msg}")
                
    except httpx.TimeoutException:
        logger.error("Stable Audio generation timed out")
        raise RuntimeError("Audio generation timed out - try a shorter duration")
    except Exception as e:
        logger.error(f"Error generating full track: {e}")
        raise


async def separate_stems_demucs(audio_bytes: bytes) -> Dict[str, bytes]:
    """
    Separate audio into instrument stems using Demucs.
    
    Uses the htdemucs model for high-quality separation into:
    - drums
    - bass
    - vocals (labeled as "melody" for instrumental tracks)
    - other (guitars, keys, etc.)
    
    Args:
        audio_bytes: Input audio as bytes (WAV format)
        
    Returns:
        Dict mapping stem names to audio bytes
    """
    try:
        import torch
        import torchaudio
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        
        # Load model (htdemucs for quality)
        model = get_model("htdemucs")
        model.eval()
        
        # Check for GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        # Load audio from bytes
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name
        
        try:
            wav, sr = torchaudio.load(temp_path)
            
            # Resample to model's expected sample rate if needed
            if sr != model.samplerate:
                wav = torchaudio.functional.resample(wav, sr, model.samplerate)
                sr = model.samplerate
            
            # Add batch dimension
            wav = wav.unsqueeze(0).to(device)
            
            # Apply model
            with torch.no_grad():
                sources = apply_model(model, wav, device=device)
            
            # Save stems
            stems = {}
            stem_names = model.sources  # ['drums', 'bass', 'other', 'vocals']
            
            for i, name in enumerate(stem_names):
                stem_audio = sources[0, i].cpu()
                
                # Convert to bytes
                buffer = io.BytesIO()
                torchaudio.save(buffer, stem_audio, sr, format="wav")
                buffer.seek(0)
                
                # Rename "vocals" to "melody" for instrumental context
                output_name = "melody" if name == "vocals" else name
                stems[output_name] = buffer.read()
            
            return stems
            
        finally:
            os.unlink(temp_path)
            
    except ImportError:
        logger.warning("Demucs not installed, using Replicate API fallback")
        return await separate_stems_replicate(audio_bytes)
    except Exception as e:
        logger.error(f"Demucs separation failed: {e}, using Replicate fallback")
        return await separate_stems_replicate(audio_bytes)


async def separate_stems_replicate(audio_bytes: bytes) -> Dict[str, bytes]:
    """
    Separate stems using Demucs via Replicate API (no local GPU required).
    """
    token = _get_replicate_token()
    if not token:
        raise ValueError("REPLICATE_API_TOKEN not set - cannot separate stems (install Demucs locally or set token)")
    
    try:
        import base64
        import asyncio
        
        # Upload audio as base64 data URI
        audio_b64 = base64.b64encode(audio_bytes).decode()
        audio_uri = f"data:audio/wav;base64,{audio_b64}"
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            # Use Demucs on Replicate
            response = await client.post(
                "https://api.replicate.com/v1/predictions",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={
                    "version": "25a173108cff36ef9f80f854c162d01df9e6528be175794b81f7b9dfb6d2f8a2",  # demucs
                    "input": {
                        "audio": audio_uri,
                        "stem": "all",  # Get all stems
                    },
                },
            )
            response.raise_for_status()
            prediction = response.json()
            
            prediction_url = prediction["urls"]["get"]
            while True:
                response = await client.get(
                    prediction_url,
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                result = response.json()
                
                if result["status"] == "succeeded":
                    output = result["output"]
                    stems = {}
                    
                    # Download each stem
                    for stem_name, stem_url in output.items():
                        if stem_url:
                            stem_response = await client.get(stem_url)
                            stem_response.raise_for_status()
                            # Rename "vocals" to "melody"
                            name = "melody" if stem_name == "vocals" else stem_name
                            stems[name] = stem_response.content
                    
                    if not stems:
                        raise RuntimeError("Demucs returned no stems")
                    return stems
                    
                elif result["status"] == "failed":
                    error_msg = result.get('error', 'Unknown error')
                    logger.error(f"Demucs separation failed: {error_msg}")
                    raise RuntimeError(f"Demucs separation failed: {error_msg}")
                    
                await asyncio.sleep(2)
                
    except Exception as e:
        logger.error(f"Error with Replicate Demucs: {e}")
        raise


# Legacy function for backward compatibility
def separate_stems(audio_bytes: bytes) -> Dict[str, bytes]:
    """Sync wrapper - prefer async separate_stems_demucs."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    return loop.run_until_complete(separate_stems_demucs(audio_bytes))
