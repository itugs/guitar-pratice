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
from .transcribe import transcribe_to_midi
from .notation import midi_to_sheet_music

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


def _generate_sheet_music(waveform: np.ndarray, sample_rate: int, output_path: str, max_duration: int) -> None:
    """
    Generate sheet music PDF from guitar waveform.

    This function implements graceful degradation - if sheet generation fails,
    it logs a warning but does NOT raise an exception (audio is already saved).

    Args:
        waveform: Original guitar waveform (before time stretch)
        sample_rate: Audio sample rate
        output_path: Path to save PDF file
        max_duration: Maximum audio duration in seconds (reject longer audio)
    """
    try:
        # Check LilyPond availability
        if not shutil.which('lilypond'):
            console.print("[yellow]⚠ LilyPond not found. Install with: brew install lilypond[/yellow]")
            console.print("[yellow]  Sheet music generation skipped.[/yellow]")
            return

        # Check audio duration
        duration = waveform.shape[1] / sample_rate  # shape is (channels, samples)
        if duration > max_duration:
            console.print(f"[yellow]⚠ Audio is {duration:.1f}s (limit: {max_duration}s)[/yellow]")
            console.print(f"[yellow]  Sheet music for long audio is impractical. Use --max-duration to override.[/yellow]")
            console.print("[yellow]  Sheet music generation skipped.[/yellow]")
            return

        console.print(f"\n[bold]Generating sheet music:[/bold]")

        # Transcribe to MIDI
        midi_path = Path(_temp_dir) / "transcription.mid"
        transcribe_to_midi(waveform, sample_rate, str(midi_path))

        # Convert MIDI to sheet music PDF
        midi_to_sheet_music(str(midi_path), output_path)

        # Success
        pdf_size = os.path.getsize(output_path) / (1024 * 1024)
        console.print(f"[green]✓ Sheet music saved: {output_path} ({pdf_size:.2f}MB)[/green]")
        console.print("[dim]  Note: Draft notation - may need cleanup in MuseScore[/dim]")

    except Exception as e:
        # Graceful degradation: log warning but don't crash
        console.print(f"[yellow]⚠ Sheet music generation failed: {e}[/yellow]")
        console.print("[yellow]  Audio was saved successfully. Sheet music skipped.[/yellow]")


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
@click.option('--sheet', type=click.Path(), default=None, help='Generate sheet music PDF at this path')
@click.option('--max-duration', type=int, default=300, help='Max audio duration for sheet generation (seconds, default: 300)')
@click.option('--keep-temp', is_flag=True, help='Keep temporary files for debugging')
def main(input: str, output: str, speed: float, bitrate: int, sheet: str, max_duration: int, keep_temp: bool):
    """
    Extract guitar track from audio files or YouTube videos.

    INPUT can be:
    - YouTube URL (e.g., https://www.youtube.com/watch?v=...)
    - Local audio file (MP3, WAV, FLAC, etc.)

    Examples:
        guitar-practice "https://youtube.com/..." -o guitar.mp3
        guitar-practice song.mp3 -o guitar.mp3 --speed 0.75
        guitar-practice song.mp3 -o guitar.mp3 --sheet music.pdf
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
        console.print(f"\n[bold]Step 2/4:[/bold] Separating guitar from mix")
        guitar_waveform, sample_rate = extract_guitar_stem(waveform, sample_rate)

        # Free memory from original waveform
        del waveform

        # Keep original guitar stem for sheet music generation (before time stretch)
        original_guitar = guitar_waveform.copy() if sheet else None

        # Step 3: Time stretch if needed
        if speed != 1.0:
            console.print(f"\n[bold]Step 3/4:[/bold] Applying speed control")
            guitar_waveform = time_stretch(guitar_waveform, sample_rate, speed)
        else:
            console.print(f"\n[bold]Step 3/4:[/bold] No speed adjustment")

        # Step 4: Export audio (FIRST - for graceful degradation)
        console.print(f"\n[bold]Step 4/4:[/bold] Exporting audio")
        export_to_mp3(guitar_waveform, sample_rate, output, bitrate)

        # Audio export complete - show success
        file_size = os.path.getsize(output) / (1024 * 1024)
        console.print(f"[green]✓ Audio saved: {output} ({file_size:.1f}MB)[/green]")

        if speed != 1.0:
            console.print(f"  Speed: {speed}x (pitch preserved)")

        # Step 5: Generate sheet music (optional, graceful degradation)
        if sheet:
            _generate_sheet_music(original_guitar, sample_rate, sheet, max_duration)

        # Final success summary
        console.print(f"\n[bold green]✓ Complete![/bold green]")

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
