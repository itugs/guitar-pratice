"""
Unit tests for audio transcription module.
"""
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from pydub import AudioSegment

from guitar_practice.transcribe import transcribe_to_midi


def test_transcribe_mono_waveform():
    """Test MIDI transcription from mono audio waveform."""
    # Load fixture
    fixture_path = Path(__file__).parent / "fixtures" / "guitar_sample.wav"
    audio = AudioSegment.from_wav(str(fixture_path))

    # Convert to numpy array (mono)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    samples = samples / (2**15)  # Normalize to [-1, 1]

    # Transcribe
    with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
        output_path = f.name

    try:
        transcribe_to_midi(samples, audio.frame_rate, output_path)

        # Verify MIDI file was created
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_transcribe_stereo_waveform():
    """Test MIDI transcription from stereo audio waveform."""
    # Load fixture
    fixture_path = Path(__file__).parent / "fixtures" / "guitar_sample.wav"
    audio = AudioSegment.from_wav(str(fixture_path))

    # Convert to stereo numpy array (2, samples)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    samples = samples / (2**15)  # Normalize to [-1, 1]

    # Duplicate to stereo
    stereo_samples = np.stack([samples, samples])

    # Transcribe
    with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
        output_path = f.name

    try:
        transcribe_to_midi(stereo_samples, audio.frame_rate, output_path)

        # Verify MIDI file was created
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_transcribe_empty_audio():
    """Test transcription fails gracefully with silent audio."""
    # Create 1 second of silence
    silent_audio = np.zeros(22050, dtype=np.float32)

    with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
        output_path = f.name

    try:
        # Should complete without error (basic-pitch may produce empty MIDI)
        transcribe_to_midi(silent_audio, 22050, output_path)

        # MIDI file should exist even if empty
        assert os.path.exists(output_path)
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_transcribe_output_matches_fixture():
    """Test that transcription produces consistent results."""
    # Load fixture audio
    fixture_audio_path = Path(__file__).parent / "fixtures" / "guitar_sample.wav"
    fixture_midi_path = Path(__file__).parent / "fixtures" / "guitar_sample.mid"

    audio = AudioSegment.from_wav(str(fixture_audio_path))
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    samples = samples / (2**15)

    # Transcribe to temp file
    with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
        output_path = f.name

    try:
        transcribe_to_midi(samples, audio.frame_rate, output_path)

        # Compare file sizes (should be similar, allowing for minor variation)
        output_size = os.path.getsize(output_path)
        fixture_size = os.path.getsize(fixture_midi_path)

        # Allow 10% variation (MIDI encoding may vary slightly)
        assert abs(output_size - fixture_size) / fixture_size < 0.1
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)
