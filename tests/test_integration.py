"""
Integration tests for end-to-end workflows.
"""
import os
import shutil
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from guitar_practice.__main__ import main


def test_local_file_to_audio_and_sheet():
    """Test complete workflow: local audio file → isolated guitar + sheet music."""
    runner = CliRunner()
    fixture_audio = Path(__file__).parent / "fixtures" / "guitar_sample.wav"

    with tempfile.TemporaryDirectory() as tmpdir:
        output_audio = os.path.join(tmpdir, "output.mp3")
        output_sheet = os.path.join(tmpdir, "output.pdf")

        # Skip if LilyPond not installed
        if not shutil.which('lilypond'):
            pytest.skip("LilyPond not installed")

        # Run CLI with sheet generation
        result = runner.invoke(
            main,
            [str(fixture_audio), '-o', output_audio, '--sheet', output_sheet]
        )

        # Should succeed
        assert result.exit_code == 0

        # Audio should be created
        assert os.path.exists(output_audio)
        assert os.path.getsize(output_audio) > 0

        # Sheet music should be created
        assert os.path.exists(output_sheet)
        assert os.path.getsize(output_sheet) > 0

        # Verify PDF header
        with open(output_sheet, 'rb') as f:
            header = f.read(4)
            assert header == b'%PDF'


def test_local_file_audio_only():
    """Test workflow without sheet music generation."""
    runner = CliRunner()
    fixture_audio = Path(__file__).parent / "fixtures" / "guitar_sample.wav"

    with tempfile.TemporaryDirectory() as tmpdir:
        output_audio = os.path.join(tmpdir, "output.mp3")

        # Run CLI without --sheet flag
        result = runner.invoke(
            main,
            [str(fixture_audio), '-o', output_audio]
        )

        # Should succeed
        assert result.exit_code == 0

        # Audio should be created
        assert os.path.exists(output_audio)
        assert os.path.getsize(output_audio) > 0


def test_local_file_with_speed_adjustment():
    """Test workflow with speed adjustment and sheet music."""
    runner = CliRunner()
    fixture_audio = Path(__file__).parent / "fixtures" / "guitar_sample.wav"

    # Skip if RubberBand not installed (required for time stretching)
    if not shutil.which('rubberband'):
        pytest.skip("RubberBand not installed")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_audio = os.path.join(tmpdir, "output.mp3")
        output_sheet = os.path.join(tmpdir, "output.pdf")

        # Skip if LilyPond not installed
        if not shutil.which('lilypond'):
            pytest.skip("LilyPond not installed")

        # Run CLI with 0.75x speed
        result = runner.invoke(
            main,
            [str(fixture_audio), '-o', output_audio, '--speed', '0.75', '--sheet', output_sheet]
        )

        # Should succeed
        assert result.exit_code == 0

        # Both outputs should exist
        assert os.path.exists(output_audio)
        assert os.path.exists(output_sheet)


def test_graceful_degradation_sheet_failure():
    """Test that audio is saved even if sheet generation fails."""
    runner = CliRunner()
    fixture_audio = Path(__file__).parent / "fixtures" / "guitar_sample.wav"

    with tempfile.TemporaryDirectory() as tmpdir:
        output_audio = os.path.join(tmpdir, "output.mp3")
        # Invalid PDF path (directory doesn't exist)
        output_sheet = "/nonexistent/directory/output.pdf"

        # Run CLI - sheet will fail but audio should succeed
        result = runner.invoke(
            main,
            [str(fixture_audio), '-o', output_audio, '--sheet', output_sheet]
        )

        # May exit with error or succeed with warning (graceful degradation)
        # Audio should still be created
        assert os.path.exists(output_audio)
        assert os.path.getsize(output_audio) > 0

        # Sheet should NOT exist
        assert not os.path.exists(output_sheet)


def test_long_audio_rejection():
    """Test that sheet generation is skipped for long audio."""
    runner = CliRunner()
    fixture_audio = Path(__file__).parent / "fixtures" / "guitar_sample.wav"

    with tempfile.TemporaryDirectory() as tmpdir:
        output_audio = os.path.join(tmpdir, "output.mp3")
        output_sheet = os.path.join(tmpdir, "output.pdf")

        # Set max-duration to 1 second (fixture is 5 seconds)
        result = runner.invoke(
            main,
            [str(fixture_audio), '-o', output_audio, '--sheet', output_sheet, '--max-duration', '1']
        )

        # Should succeed
        assert result.exit_code == 0

        # Audio should be created
        assert os.path.exists(output_audio)

        # Sheet should NOT be created (duration exceeded)
        assert not os.path.exists(output_sheet)

        # Output should mention duration limit
        assert 'Audio is' in result.output or 'limit' in result.output.lower()
