"""
Audio processing module - time stretching with pitch preservation.
"""
import subprocess
import platform

import numpy as np
import pyrubberband as pyrb
from rich.console import Console

console = Console()


def check_rubberband() -> None:
    """Check if RubberBand library is installed, raise with install hint if missing."""
    try:
        # Try importing pyrubberband - it will fail if RubberBand C++ lib is missing
        pyrb.pyrb.get_rubberband_version()
    except Exception:
        system = platform.system()
        if system == "Darwin":
            hint = "brew install rubberband"
        elif system == "Linux":
            hint = "apt install rubberband-cli  # or: yum install rubberband"
        else:
            hint = "Download from https://breakfastquay.com/rubberband/"
        raise SystemExit(f"RubberBand library not found. Install it: {hint}")


def time_stretch(waveform: np.ndarray, sample_rate: int, rate: float) -> np.ndarray:
    """
    Time-stretch audio while preserving pitch using RubberBand.

    Args:
        waveform: Audio waveform, shape (channels, samples)
        sample_rate: Audio sample rate
        rate: Time stretch factor (e.g., 0.75 = 75% speed, 1.25 = 125% speed)

    Returns:
        Time-stretched waveform with same shape as input

    Raises:
        SystemExit: If rate is out of bounds or RubberBand is missing
    """
    # Validate rate
    if rate < 0.25 or rate > 4.0:
        raise SystemExit(
            f"Speed rate {rate} is out of bounds. "
            f"Valid range: 0.25 (very slow) to 4.0 (very fast)"
        )

    # Warn about quality at extreme rates
    if rate < 0.5 or rate > 1.5:
        console.print(
            f"[yellow]⚠ Warning: Extreme speed change (rate={rate}) may introduce artifacts[/yellow]"
        )

    # Check RubberBand is installed
    check_rubberband()

    try:
        console.print(f"[cyan]Time-stretching audio to {rate}x speed (pitch preserved)...[/cyan]")

        # pyrubberband expects shape (samples, channels) for stereo
        # or (samples,) for mono
        if waveform.shape[0] == 1:
            # Mono: squeeze to (samples,)
            audio_in = waveform.squeeze()
        else:
            # Stereo/multi-channel: transpose to (samples, channels)
            audio_in = waveform.T

        # Apply time stretch
        stretched = pyrb.time_stretch(audio_in, sample_rate, rate)

        # Convert back to (channels, samples)
        if stretched.ndim == 1:
            # Mono: expand to (1, samples)
            stretched = stretched[np.newaxis, :]
        else:
            # Multi-channel: transpose back
            stretched = stretched.T

        console.print(f"[green]✓ Time-stretch complete[/green]")
        return stretched

    except Exception as e:
        raise SystemExit(f"Time-stretch failed: {e}")
