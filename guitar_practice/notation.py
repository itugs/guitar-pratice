"""
Music notation module - music21 integration for sheet music generation.
"""
from pathlib import Path

import music21
from rich.console import Console

console = Console()


def midi_to_sheet_music(midi_path: str, output_pdf_path: str) -> None:
    """
    Convert MIDI to sheet music PDF with guitar notation semantics.

    Args:
        midi_path: Path to MIDI file (from basic-pitch)
        output_pdf_path: Path to save PDF file

    Raises:
        SystemExit: On conversion or rendering failure

    Note:
        - Applies treble 8vb clef (guitar sounds octave lower)
        - Uses music21's built-in LilyPond export (no custom subprocess)
        - File naming quirk: music21 creates output.pdf.pdf (handled automatically)
    """
    try:
        console.print("[cyan]Converting MIDI to sheet music...[/cyan]")

        # Load MIDI into music21
        score = music21.converter.parse(midi_path)

        if not score.parts:
            raise ValueError("MIDI file contains no musical parts")

        # Apply guitar notation semantics
        _apply_guitar_notation(score)

        console.print("[cyan]Rendering PDF with LilyPond...[/cyan]")

        # Export to PDF using music21's built-in LilyPond export
        # Note: music21 writes .ly source as "file.pdf", then LilyPond creates "file.pdf.pdf"
        # We handle this by using a temp path, then renaming the actual PDF
        temp_base = Path(output_pdf_path).with_suffix('')  # Remove .pdf extension

        # music21.write() will create temp_base.pdf (the .ly source)
        # and LilyPond will create temp_base.pdf.pdf (the actual PDF)
        score.write('lily.pdf', fp=str(temp_base))

        # Find the actual PDF (it will have .pdf.pdf extension)
        actual_pdf = Path(str(temp_base) + '.pdf.pdf')

        if not actual_pdf.exists():
            # Fallback: sometimes it's just .pdf
            actual_pdf = Path(str(temp_base) + '.pdf')
            if not actual_pdf.exists():
                raise FileNotFoundError("LilyPond did not create PDF output")

        # Rename to expected output path
        actual_pdf.rename(output_pdf_path)

        # Clean up .ly source if it exists
        ly_source = Path(str(temp_base) + '.pdf')  # The .ly file music21 created
        if ly_source.exists() and ly_source != Path(output_pdf_path):
            ly_source.unlink()

        console.print(f"[green]✓ Sheet music PDF created: {output_pdf_path}[/green]")

    except Exception as e:
        raise SystemExit(f"Sheet music generation failed: {e}")


def _apply_guitar_notation(score: music21.stream.Score) -> None:
    """
    Apply guitar-specific notation semantics to a music21 Score.

    Modifications:
    1. Set treble 8vb clef (guitar sounds octave lower than written)
    2. Use music21's default enharmonic spelling (key-aware)
    3. Single-voice output (basic-pitch produces single-voice MIDI)
    """
    for part in score.parts:
        # 1. Apply treble 8vb clef
        # Remove any existing clefs and insert treble 8vb at the beginning
        for clef in part.flatten().getElementsByClass(music21.clef.Clef):
            part.remove(clef)

        part.insert(0, music21.clef.Treble8vbClef())

        # 2. Enharmonic spelling: music21 handles this automatically based on key context
        # No action needed - defaults are appropriate

        # 3. Voice separation: not needed - basic-pitch outputs single-voice MIDI

    console.print("[green]✓ Applied guitar notation (treble 8vb clef)[/green]")
