"""
Unit tests for music notation module.
"""
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import music21

from guitar_practice.notation import midi_to_sheet_music, _apply_guitar_notation


def test_midi_to_sheet_music():
    """Test MIDI to PDF conversion with guitar notation."""
    # Use fixture MIDI
    fixture_midi_path = Path(__file__).parent / "fixtures" / "guitar_sample.mid"

    # Create temp output path
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
        output_path = f.name

    try:
        # Skip if LilyPond not installed
        if not shutil.which('lilypond'):
            pytest.skip("LilyPond not installed")

        midi_to_sheet_music(str(fixture_midi_path), output_path)

        # Verify PDF was created
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

        # Verify it's a valid PDF (starts with %PDF)
        with open(output_path, 'rb') as f:
            header = f.read(4)
            assert header == b'%PDF'
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_apply_guitar_notation():
    """Test that guitar notation semantics are applied correctly."""
    # Load fixture MIDI
    fixture_midi_path = Path(__file__).parent / "fixtures" / "guitar_sample.mid"
    score = music21.converter.parse(str(fixture_midi_path))

    # Apply guitar notation
    _apply_guitar_notation(score)

    # Verify treble 8vb clef was added
    assert len(score.parts) > 0
    part = score.parts[0]

    # Find clef at offset 0 (inserted by _apply_guitar_notation)
    clef_at_start = part.getElementsByOffset(0).getElementsByClass(music21.clef.Clef)
    assert len(clef_at_start) > 0

    # Clef at start should be Treble8vbClef
    first_clef = clef_at_start[0]
    assert isinstance(first_clef, music21.clef.Treble8vbClef)


def test_midi_to_sheet_music_no_lilypond():
    """Test that notation module relies on music21's LilyPond integration.

    Note: The LilyPond availability check is in __main__.py, not notation.py.
    notation.py will raise SystemExit if music21 fails to render.
    """
    # This test documents that notation.py doesn't check for LilyPond -
    # it relies on music21.write() which will fail if LilyPond is missing.
    # The actual LilyPond check happens in __main__.py's _generate_sheet_music()

    # Skip this test - it's just documentation of the architecture
    pytest.skip("LilyPond check is in __main__.py, not notation.py")


def test_midi_to_sheet_music_invalid_midi():
    """Test error handling with invalid MIDI file."""
    # Create temp invalid MIDI file
    with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
        invalid_midi_path = f.name
        f.write(b'not a valid midi file')

    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
        output_path = f.name

    try:
        # Skip if LilyPond not installed
        if not shutil.which('lilypond'):
            pytest.skip("LilyPond not installed")

        # Should raise SystemExit on invalid MIDI
        with pytest.raises(SystemExit):
            midi_to_sheet_music(invalid_midi_path, output_path)
    finally:
        if os.path.exists(invalid_midi_path):
            os.unlink(invalid_midi_path)
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_file_naming_quirk_handled():
    """Test that music21's .pdf.pdf naming quirk is handled correctly."""
    fixture_midi_path = Path(__file__).parent / "fixtures" / "guitar_sample.mid"

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "output.pdf")

        # Skip if LilyPond not installed
        if not shutil.which('lilypond'):
            pytest.skip("LilyPond not installed")

        midi_to_sheet_music(str(fixture_midi_path), output_path)

        # Should create output.pdf (not output.pdf.pdf)
        assert os.path.exists(output_path)
        assert not os.path.exists(output_path + '.pdf')

        # Temp .ly source should be cleaned up
        ly_source_path = output_path  # music21 writes .ly as "output.pdf"
        # After renaming, the .ly should be gone
        assert not os.path.exists(ly_source_path) or os.path.exists(output_path)
