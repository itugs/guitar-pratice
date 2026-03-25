"""
Audio transcription module - basic-pitch integration for guitar MIDI generation.
"""
import tempfile
import os
import numpy as np
import soundfile as sf
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
from rich.console import Console

console = Console()


def transcribe_to_midi(waveform: np.ndarray, sample_rate: int, output_path: str) -> None:
    """
    Transcribe guitar audio to MIDI using basic-pitch.

    Args:
        waveform: Audio waveform as numpy array, shape (channels, samples)
        sample_rate: Audio sample rate (typically 44100Hz from Demucs)
        output_path: Path to save MIDI file

    Raises:
        SystemExit: On transcription failure

    Note:
        basic-pitch uses CoreML automatically on Mac - no device detection needed.
        basic-pitch will resample audio to 22050Hz internally if needed.
    """
    temp_audio_path = None
    try:
        console.print("[cyan]Transcribing audio to MIDI with basic-pitch...[/cyan]")

        # basic-pitch predict() expects a file path, not audio data
        # Save waveform to temp file
        temp_fd, temp_audio_path = tempfile.mkstemp(suffix='.wav')
        os.close(temp_fd)  # Close the file descriptor, we'll write with soundfile

        # Convert waveform to format expected by soundfile
        # soundfile expects (samples, channels) for stereo, (samples,) for mono
        if waveform.ndim == 1:
            audio_data = waveform
        elif waveform.shape[0] == 1:
            # Mono stored as (1, samples) → squeeze to (samples,)
            audio_data = waveform.squeeze()
        else:
            # Stereo (2, samples) → transpose to (samples, 2)
            audio_data = waveform.T

        sf.write(temp_audio_path, audio_data, sample_rate)

        # basic-pitch predict returns: model_output, midi_data, note_events
        _, midi_data, note_events = predict(
            temp_audio_path,
            model_or_model_path=ICASSP_2022_MODEL_PATH
        )

        # Save MIDI
        midi_data.write(output_path)

        console.print(f"[green]✓ MIDI transcription complete: {len(note_events)} notes detected[/green]")

    except Exception as e:
        raise SystemExit(f"MIDI transcription failed: {e}")

    finally:
        # Clean up temp file
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except Exception:
                pass  # Ignore cleanup errors
