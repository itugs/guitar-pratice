"""
Audio download module - handles YouTube URLs and local files.
"""
import os
import tempfile
import subprocess
import platform
from pathlib import Path
from typing import Tuple

import yt_dlp
import librosa
import soundfile as sf
import numpy as np
from rich.console import Console

console = Console()


def check_ffmpeg() -> None:
    """Check if FFmpeg is installed, raise with install hint if missing."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except FileNotFoundError:
        system = platform.system()
        if system == "Darwin":
            hint = "brew install ffmpeg"
        elif system == "Linux":
            hint = "apt install ffmpeg  # or: yum install ffmpeg"
        else:
            hint = "Download from https://ffmpeg.org"
        raise SystemExit(f"FFmpeg not found. Install it: {hint}")


def is_youtube_url(input_str: str) -> bool:
    """Check if input is a YouTube URL."""
    return any(domain in input_str for domain in ['youtube.com', 'youtu.be'])


def download_youtube_audio(url: str, output_dir: str) -> Tuple[np.ndarray, int]:
    """
    Download audio from YouTube URL.

    Args:
        url: YouTube video URL
        output_dir: Directory to save temporary audio file

    Returns:
        Tuple of (waveform as numpy array, sample rate)

    Raises:
        SystemExit: On download failure after retries
    """
    output_path = os.path.join(output_dir, 'downloaded_audio.wav')

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'outtmpl': output_path.replace('.wav', ''),
        'quiet': False,
        'no_warnings': False,
        'retries': 3,
        'fragment_retries': 3,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            console.print(f"[cyan]Downloading audio from YouTube...[/cyan]")
            ydl.download([url])

        # Load the downloaded audio
        waveform, sr = librosa.load(output_path, sr=None, mono=False)

        # Ensure waveform is 2D (channels, samples)
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]

        return waveform, sr

    except yt_dlp.utils.DownloadError as e:
        error_msg = str(e)
        if 'HTTP Error 429' in error_msg:
            raise SystemExit("YouTube rate limit reached. Try again later.")
        elif 'age' in error_msg.lower() and 'restrict' in error_msg.lower():
            raise SystemExit("Age-restricted video. Cannot download without authentication.")
        else:
            raise SystemExit(f"Download failed: {error_msg}")
    except Exception as e:
        raise SystemExit(f"Unexpected error during download: {e}")


def load_local_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """
    Load audio from local file.

    Args:
        file_path: Path to local audio file

    Returns:
        Tuple of (waveform as numpy array, sample rate)

    Raises:
        SystemExit: If file not found or format unsupported
    """
    if not os.path.exists(file_path):
        raise SystemExit(f"File not found: {file_path}")

    try:
        waveform, sr = librosa.load(file_path, sr=None, mono=False)

        # Ensure waveform is 2D (channels, samples)
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]

        console.print(f"[green]Loaded audio from {file_path}[/green]")
        return waveform, sr

    except Exception as e:
        supported_formats = "MP3, WAV, FLAC, OGG, M4A, AAC"
        raise SystemExit(
            f"Failed to load audio file: {e}\n"
            f"Supported formats: {supported_formats}"
        )


def download_audio(input_source: str) -> Tuple[np.ndarray, int]:
    """
    Download or load audio from YouTube URL or local file.

    Args:
        input_source: YouTube URL or path to local audio file

    Returns:
        Tuple of (waveform as numpy array, sample rate)
    """
    # Check FFmpeg first
    check_ffmpeg()

    if is_youtube_url(input_source):
        temp_dir = tempfile.mkdtemp(prefix='guitar-practice-')
        try:
            return download_youtube_audio(input_source, temp_dir)
        finally:
            # Cleanup temp download directory
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass  # Ignore cleanup errors
    else:
        return load_local_audio(input_source)
