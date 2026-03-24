"""
CLI entry point for guitar-practice tool.
"""
import os
import sys
import tempfile
import shutil
import signal
from pathlib import Path

import click
import numpy as np
import torch
from pydub import AudioSegment
from demucs.audio import save_audio
from rich.console import Console

from .download import download_audio
from .separate import extract_guitar_stem
from .process import time_stretch

console = Console()

# Global temp directory for cleanup on interrupt
_temp_dir = None


def cleanup_temp_files():
    """Clean up temporary files."""
    global _temp_dir
    if _temp_dir and os.path.exists(_temp_dir):
        try:
            shutil.rmtree(_temp_dir)
        except Exception:
            pass  # Ignore cleanup errors


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    console.print("\n[yellow]Interrupted by user. Cleaning up...[/yellow]")
    cleanup_temp_files()
    sys.exit(1)


def check_disk_space(required_mb: int = 500) -> None:
    """
    Check if sufficient disk space is available.

    Args:
        required_mb: Required disk space in megabytes

    Raises:
        SystemExit: If insufficient disk space
    """
    stat = shutil.disk_usage(tempfile.gettempdir())
    available_mb = stat.free // (1024 * 1024)

    if available_mb < required_mb:
        raise SystemExit(
            f"Insufficient disk space: {available_mb}MB available, need ~{required_mb}MB\n"
            f"Free up space in {tempfile.gettempdir()} and try again."
        )


def export_to_mp3(waveform: np.ndarray, sample_rate: int, output_path: str, bitrate: int = 192) -> None:
    """
    Export waveform to MP3 file using Demucs's built-in save_audio.

    Args:
        waveform: Audio waveform, shape (channels, samples)
        sample_rate: Audio sample rate
        output_path: Path to output MP3 file
        bitrate: MP3 bitrate in kbps

    Raises:
        SystemExit: On export failure
    """
    try:
        console.print(f"[cyan]Exporting to MP3 ({bitrate}kbps)...[/cyan]")

        # Convert numpy to torch tensor
        audio_tensor = torch.from_numpy(waveform).float()

        # Use Demucs's save_audio for proper float32 → MP3 conversion
        save_audio(audio_tensor, output_path, sample_rate, bitrate=bitrate)

        console.print(f"[green]✓ Saved to {output_path}[/green]")

    except Exception as e:
        # Check if disk full
        if "No space left" in str(e) or "Disk full" in str(e):
            raise SystemExit("Disk full during MP3 export. Free up space and try again.")
        raise SystemExit(f"MP3 export failed: {e}")


@click.command()
@click.argument('input', type=str)
@click.option('-o', '--output', required=True, type=click.Path(), help='Output MP3 file path')
@click.option('--speed', type=float, default=1.0, help='Playback speed (0.25-4.0, default: 1.0)')
@click.option('--bitrate', type=int, default=192, help='MP3 bitrate in kbps (default: 192)')
@click.option('--keep-temp', is_flag=True, help='Keep temporary files for debugging')
def main(input: str, output: str, speed: float, bitrate: int, keep_temp: bool):
    """
    Extract guitar track from audio files or YouTube videos.

    INPUT can be:
    - YouTube URL (e.g., https://www.youtube.com/watch?v=...)
    - Local audio file (MP3, WAV, FLAC, etc.)

    Examples:
        guitar-practice "https://youtube.com/..." -o guitar.mp3
        guitar-practice song.mp3 -o guitar.mp3 --speed 0.75
    """
    global _temp_dir

    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Startup checks
        console.print("[bold cyan]Guitar Practice Suite v0.1.0[/bold cyan]\n")

        check_disk_space()

        # Create temp directory
        _temp_dir = tempfile.mkdtemp(prefix='guitar-practice-')

        # Step 1: Download/load audio
        console.print(f"[bold]Step 1/3:[/bold] Loading audio from {input}")
        waveform, sample_rate = download_audio(input)

        # Step 2: Separate stems and extract guitar
        console.print(f"\n[bold]Step 2/3:[/bold] Separating guitar from mix")
        guitar_waveform, sample_rate = extract_guitar_stem(waveform, sample_rate)

        # Free memory from original waveform
        del waveform

        # Step 3: Time stretch if needed
        if speed != 1.0:
            console.print(f"\n[bold]Step 3/3:[/bold] Applying speed control")
            guitar_waveform = time_stretch(guitar_waveform, sample_rate, speed)
        else:
            console.print(f"\n[bold]Step 3/3:[/bold] No speed adjustment")

        # Step 4: Export to MP3
        console.print(f"\n[bold]Exporting:[/bold]")
        export_to_mp3(guitar_waveform, sample_rate, output, bitrate)

        # Success summary
        console.print(f"\n[bold green]✓ Success![/bold green]")
        file_size = os.path.getsize(output) / (1024 * 1024)
        console.print(f"Output file: {output} ({file_size:.1f}MB)")

        if speed != 1.0:
            console.print(f"Speed: {speed}x (pitch preserved)")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(1)

    except SystemExit as e:
        # Re-raise SystemExit (these are intentional error messages)
        raise

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        sys.exit(1)

    finally:
        # Cleanup temp files unless --keep-temp
        if not keep_temp:
            cleanup_temp_files()


if __name__ == '__main__':
    main()
