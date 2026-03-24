"""
Audio separation module - Demucs integration for guitar isolation.
"""
from typing import Dict, Tuple

import torch
import numpy as np
from demucs import pretrained
from demucs.apply import apply_model
from demucs.audio import convert_audio
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()


def detect_device() -> str:
    """
    Detect best available device (CUDA, MPS, or CPU).

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        console.print("[green]✓ Using NVIDIA GPU (CUDA)[/green]")
        console.print("[dim]Expected processing time: 1-2 min per song[/dim]")
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        console.print("[green]✓ Using Apple Silicon GPU (MPS)[/green]")
        console.print("[dim]Expected processing time: 1-2 min per song[/dim]")
        return "mps"
    else:
        console.print("[yellow]⚠ No GPU detected - using CPU[/yellow]")
        console.print("[yellow]WARNING: CPU processing takes 5-10 min per song[/yellow]")
        console.print("[dim]Tip: For faster processing, use a machine with CUDA or Apple Silicon GPU[/dim]")
        return "cpu"


def separate_stems(waveform: np.ndarray, sample_rate: int, device: str = None) -> Dict[str, np.ndarray]:
    """
    Separate audio into stems using Demucs.

    Args:
        waveform: Audio waveform as numpy array, shape (channels, samples)
        sample_rate: Audio sample rate
        device: Device to use ('cuda', 'mps', 'cpu'). Auto-detects if None.

    Returns:
        Dictionary mapping stem names to waveforms:
        {'vocals', 'drums', 'bass', 'guitar', 'piano', 'other'}

    Raises:
        SystemExit: On out-of-memory or other Demucs errors
    """
    if device is None:
        device = detect_device()

    try:
        # Load Demucs model
        console.print("\n[cyan]Loading Demucs model...[/cyan]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Loading htdemucs_6s model...", total=None)

            # Load 6-stem model (vocals, drums, bass, guitar, piano, other)
            model = pretrained.get_model('htdemucs_6s')
            model.to(device)
            model.eval()

            progress.update(task, completed=True)

        # Convert numpy array to torch tensor
        # Demucs expects shape: (batch, channels, samples)
        audio_tensor = torch.from_numpy(waveform).float()

        # Ensure correct shape
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
        elif audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.unsqueeze(0)  # (1, channels, samples)

        # Convert audio to model's expected sample rate and channels
        # Demucs htdemucs_6s expects 44.1kHz stereo
        audio_tensor = convert_audio(audio_tensor, sample_rate, model.samplerate, model.audio_channels)
        audio_tensor = audio_tensor.to(device)

        # Apply model with progress
        console.print("[cyan]Separating audio into stems...[/cyan]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            task = progress.add_task("Processing with Demucs...", total=100)

            with torch.no_grad():
                stems = apply_model(model, audio_tensor)

            progress.update(task, completed=100)

        # Convert stems to numpy arrays
        # stems shape: (batch, stems, channels, samples)
        stems_np = stems.squeeze(0).cpu().numpy()

        # Map to stem names (htdemucs_6s order)
        stem_names = ['drums', 'bass', 'other', 'vocals', 'guitar', 'piano']
        stem_dict = {name: stems_np[i] for i, name in enumerate(stem_names)}

        console.print("[green]✓ Separation complete[/green]")
        return stem_dict

    except torch.cuda.OutOfMemoryError:
        console.print("[yellow]⚠ GPU out of memory - falling back to CPU[/yellow]")
        # Fallback to CPU
        if device != "cpu":
            return separate_stems(waveform, sample_rate, device="cpu")
        else:
            raise SystemExit("Out of memory even on CPU. Try with a shorter audio file.")

    except Exception as e:
        raise SystemExit(f"Separation failed: {e}")


def extract_guitar_stem(waveform: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, int]:
    """
    Extract only the guitar stem from audio.

    Args:
        waveform: Audio waveform as numpy array
        sample_rate: Audio sample rate

    Returns:
        Tuple of (guitar waveform, Demucs output sample rate = 44100)
    """
    # Separate all stems
    stems = separate_stems(waveform, sample_rate)

    # Extract guitar stem
    guitar = stems['guitar']

    # Free memory from other stems
    del stems

    console.print("[green]✓ Guitar stem extracted[/green]")
    # Return Demucs output sample rate (always 44.1kHz for htdemucs_6s)
    return guitar, 44100
