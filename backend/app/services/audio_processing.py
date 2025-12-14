"""Audio processing utilities for tempo alignment and cohesion.

Provides pre-processing (before Stable Audio) and post-processing (after)
to ensure generated tracks fit with existing song structure.
"""

import io
import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def detect_bpm(audio_bytes: bytes) -> Optional[float]:
    """
    Detect tempo (BPM) of audio using librosa.
    
    Args:
        audio_bytes: WAV audio as bytes
        
    Returns:
        Detected BPM or None if detection fails
    """
    try:
        import librosa
        import soundfile as sf
        
        audio_buffer = io.BytesIO(audio_bytes)
        y, sr = sf.read(audio_buffer)
        
        # Convert to mono if stereo
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        
        # Detect tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        if hasattr(tempo, '__len__'):
            tempo = float(tempo[0]) if len(tempo) > 0 else None
        else:
            tempo = float(tempo)
            
        logger.info(f"Detected BPM: {tempo:.1f}")
        return tempo
        
    except ImportError:
        logger.warning("librosa not installed - BPM detection unavailable")
        return None
    except Exception as e:
        logger.error(f"BPM detection failed: {e}")
        return None


def time_stretch_audio(
    audio_bytes: bytes,
    target_bpm: float,
    source_bpm: Optional[float] = None,
) -> bytes:
    """
    Time-stretch audio to match target BPM.
    
    Preserves pitch while changing tempo.
    
    Args:
        audio_bytes: Input WAV audio
        target_bpm: Desired output BPM
        source_bpm: Source BPM (auto-detected if None)
        
    Returns:
        Time-stretched WAV audio
    """
    try:
        import librosa
        import soundfile as sf
        
        # Load audio
        audio_buffer = io.BytesIO(audio_bytes)
        y, sr = sf.read(audio_buffer)
        
        # Handle stereo
        is_stereo = len(y.shape) > 1
        if is_stereo:
            y_mono = np.mean(y, axis=1)
        else:
            y_mono = y
        
        # Detect source BPM if not provided
        if source_bpm is None:
            tempo, _ = librosa.beat.beat_track(y=y_mono, sr=sr)
            if hasattr(tempo, '__len__'):
                source_bpm = float(tempo[0]) if len(tempo) > 0 else 120.0
            else:
                source_bpm = float(tempo)
            logger.info(f"Auto-detected source BPM: {source_bpm:.1f}")
        
        # Calculate stretch ratio
        # If source is 100 BPM and target is 120 BPM, we need to speed up (ratio > 1)
        ratio = target_bpm / source_bpm
        
        if abs(ratio - 1.0) < 0.02:  # Within 2%, skip stretching
            logger.info(f"BPM close enough ({source_bpm:.1f} → {target_bpm:.1f}), skipping stretch")
            return audio_bytes
        
        logger.info(f"Time-stretching: {source_bpm:.1f} → {target_bpm:.1f} BPM (ratio: {ratio:.3f})")
        
        # Time stretch
        if is_stereo:
            # Process each channel
            y_stretched_l = librosa.effects.time_stretch(y[:, 0], rate=ratio)
            y_stretched_r = librosa.effects.time_stretch(y[:, 1], rate=ratio)
            y_stretched = np.column_stack([y_stretched_l, y_stretched_r])
        else:
            y_stretched = librosa.effects.time_stretch(y_mono, rate=ratio)
        
        # Write to bytes
        output_buffer = io.BytesIO()
        sf.write(output_buffer, y_stretched, sr, format='WAV')
        output_buffer.seek(0)
        
        return output_buffer.read()
        
    except ImportError:
        logger.warning("librosa not installed - returning original audio")
        return audio_bytes
    except Exception as e:
        logger.error(f"Time stretch failed: {e}")
        return audio_bytes


def trim_audio_to_duration(audio_bytes: bytes, target_duration: float) -> bytes:
    """
    Trim or pad audio to exact duration.
    
    Args:
        audio_bytes: Input WAV audio
        target_duration: Desired duration in seconds
        
    Returns:
        Trimmed/padded WAV audio
    """
    try:
        import soundfile as sf
        
        audio_buffer = io.BytesIO(audio_bytes)
        y, sr = sf.read(audio_buffer)
        
        target_samples = int(target_duration * sr)
        current_samples = y.shape[0]
        
        if current_samples == target_samples:
            return audio_bytes
        
        is_stereo = len(y.shape) > 1
        
        if current_samples > target_samples:
            # Trim
            y_trimmed = y[:target_samples]
            logger.info(f"Trimmed audio: {current_samples/sr:.2f}s → {target_duration:.2f}s")
        else:
            # Pad with silence
            if is_stereo:
                padding = np.zeros((target_samples - current_samples, y.shape[1]))
            else:
                padding = np.zeros(target_samples - current_samples)
            y_trimmed = np.concatenate([y, padding])
            logger.info(f"Padded audio: {current_samples/sr:.2f}s → {target_duration:.2f}s")
        
        output_buffer = io.BytesIO()
        sf.write(output_buffer, y_trimmed, sr, format='WAV')
        output_buffer.seek(0)
        
        return output_buffer.read()
        
    except ImportError:
        logger.warning("soundfile not installed - returning original audio")
        return audio_bytes
    except Exception as e:
        logger.error(f"Trim failed: {e}")
        return audio_bytes


def get_audio_duration(audio_bytes: bytes) -> Optional[float]:
    """Get duration of audio in seconds."""
    try:
        import soundfile as sf
        
        audio_buffer = io.BytesIO(audio_bytes)
        y, sr = sf.read(audio_buffer)
        return len(y) / sr
        
    except Exception as e:
        logger.error(f"Could not get duration: {e}")
        return None


def preprocess_user_recording(
    audio_bytes: bytes,
    target_bpm: float,
) -> Tuple[bytes, Optional[float]]:
    """
    Pre-process user recording before sending to Stable Audio.
    
    - Detects source BPM
    - Time-stretches to target BPM
    
    Args:
        audio_bytes: User's recording (WAV)
        target_bpm: Target tempo to match existing tracks
        
    Returns:
        Tuple of (processed_audio, detected_source_bpm)
    """
    source_bpm = detect_bpm(audio_bytes)
    
    if source_bpm is None:
        logger.warning("Could not detect BPM, using recording as-is")
        return audio_bytes, None
    
    processed = time_stretch_audio(audio_bytes, target_bpm, source_bpm)
    return processed, source_bpm


def postprocess_generated_audio(
    audio_bytes: bytes,
    target_bpm: float,
    target_duration: float,
    max_bpm_drift: float = 10.0,
) -> Tuple[bytes, dict]:
    """
    Post-process Stable Audio output to fit existing track constraints.
    
    - Detects output BPM
    - Time-stretches if drifted from target
    - Trims to exact duration
    
    Args:
        audio_bytes: Generated audio from Stable Audio
        target_bpm: Desired BPM
        target_duration: Desired duration in seconds
        max_bpm_drift: Max acceptable BPM difference before stretching
        
    Returns:
        Tuple of (processed_audio, metadata_dict)
    """
    metadata = {
        "detected_bpm": None,
        "bpm_drift": None,
        "was_stretched": False,
        "was_trimmed": False,
        "final_duration": None,
    }
    
    # Detect output BPM
    detected_bpm = detect_bpm(audio_bytes)
    metadata["detected_bpm"] = detected_bpm
    
    processed = audio_bytes
    
    # Time-stretch if needed
    if detected_bpm is not None:
        drift = abs(detected_bpm - target_bpm)
        metadata["bpm_drift"] = drift
        
        if drift > max_bpm_drift:
            logger.info(f"BPM drift {drift:.1f} exceeds threshold, stretching...")
            processed = time_stretch_audio(processed, target_bpm, detected_bpm)
            metadata["was_stretched"] = True
    
    # Trim to exact duration
    current_duration = get_audio_duration(processed)
    if current_duration and abs(current_duration - target_duration) > 0.1:
        processed = trim_audio_to_duration(processed, target_duration)
        metadata["was_trimmed"] = True
    
    metadata["final_duration"] = get_audio_duration(processed)
    
    return processed, metadata
