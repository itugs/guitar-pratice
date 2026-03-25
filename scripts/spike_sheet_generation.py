#!/usr/bin/env python3
"""
Spike script to validate the sheet music generation pipeline.

This script tests:
- basic-pitch: audio → MIDI transcription
- music21: MIDI → Score conversion with guitar notation semantics
- LilyPond: Score → PDF rendering (via music21's built-in export)

Usage:
    python scripts/spike_sheet_generation.py [--debug]

    --debug: Keep intermediate files (MIDI, .ly) in spike_output/
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
import music21


def main():
    parser = argparse.ArgumentParser(description="Spike: Guitar audio → sheet music pipeline")
    parser.add_argument("--debug", action="store_true", help="Keep intermediate files in spike_output/")
    args = parser.parse_args()

    # Setup output directory
    output_dir = Path("spike_output")
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("SPIKE: Guitar Sheet Music Generation Pipeline")
    print("=" * 70)

    # Step 1: Find or create test audio
    print("\n[Step 1/4] Loading test audio...")

    # TODO: Use a real guitar audio file for testing
    # For now, create a simple test tone (guitar A note at 110Hz)
    sample_rate = 22050  # basic-pitch expects 22050Hz
    duration = 3.0  # 3 seconds
    frequency = 110.0  # A2 note

    t = np.linspace(0, duration, int(sample_rate * duration))
    # Simple sine wave for testing
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

    test_audio_path = output_dir / "test_guitar.wav"

    # Write test audio using soundfile
    import soundfile as sf
    sf.write(test_audio_path, audio, sample_rate)

    print(f"✓ Test audio created: {test_audio_path}")
    print(f"  Duration: {duration}s, Sample rate: {sample_rate}Hz")

    # Step 2: basic-pitch transcription (audio → MIDI)
    print("\n[Step 2/4] Transcribing with basic-pitch...")
    print("  Note: basic-pitch uses CoreML on Mac (no device detection needed)")

    midi_path = output_dir / "transcription.mid"

    try:
        # basic-pitch predict returns: model_output, midi_data, note_events
        model_output, midi_data, note_events = predict(
            str(test_audio_path),
            model_or_model_path=ICASSP_2022_MODEL_PATH
        )

        # Save MIDI
        midi_data.write(str(midi_path))

        print(f"✓ MIDI transcription complete: {midi_path}")
        print(f"  Notes detected: {len(note_events)}")

    except Exception as e:
        print(f"✗ basic-pitch transcription failed: {e}")
        sys.exit(1)

    # Step 3: music21 Score generation with guitar notation semantics
    print("\n[Step 3/4] Creating music21 Score...")

    try:
        # Load MIDI into music21
        score = music21.converter.parse(str(midi_path))

        # CRITICAL: Apply guitar notation semantics
        # 1. Set treble 8vb clef (guitar sounds octave lower)
        if score.parts:
            part = score.parts[0]
            # Remove existing clef and add treble 8vb
            for element in part.flatten().getElementsByClass(music21.clef.Clef):
                part.remove(element)
            part.insert(0, music21.clef.Treble8vbClef())
            print("✓ Applied treble 8vb clef")

        # 2. Check enharmonic spelling
        # music21 auto-spells based on key context - we'll validate this in the PDF
        print("✓ Enharmonic spelling: using music21 defaults (key-aware)")

        # 3. Check voice separation for chords
        # basic-pitch outputs single-voice MIDI, so no separation needed for spike
        print("✓ Voice separation: single voice (basic-pitch output)")

        # 4. Validate auto-barring
        # music21 automatically infers measures from MIDI note timing
        measures = list(score.flatten().getElementsByClass(music21.stream.Measure))
        if measures:
            print(f"✓ Auto-barring: {len(measures)} measures detected")
            # Check if time signature was inferred
            time_sigs = list(score.flatten().getElementsByClass(music21.meter.TimeSignature))
            if time_sigs:
                print(f"  Time signature: {time_sigs[0]}")
            else:
                print("  Time signature: not detected, using defaults")
        else:
            print("⚠ No measures detected - PDF may not have barlines")

    except Exception as e:
        print(f"✗ music21 Score creation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 4: Export to PDF using music21's built-in LilyPond export
    print("\n[Step 4/4] Rendering PDF with LilyPond...")

    pdf_path = output_dir / "sheet_music.pdf"

    try:
        # music21's built-in write method handles LilyPond subprocess
        # This is simpler than custom render.py (Codex finding #10)
        score.write('lily.pdf', fp=str(pdf_path))

        print(f"✓ PDF generated: {pdf_path}")

        # Check if PDF was created
        if pdf_path.exists():
            size_mb = pdf_path.stat().st_size / (1024 * 1024)
            print(f"  File size: {size_mb:.2f} MB")
        else:
            raise FileNotFoundError("PDF was not created")

    except Exception as e:
        print(f"✗ PDF rendering failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Cleanup intermediate files unless --debug
    print("\n[Cleanup]")
    if not args.debug:
        if test_audio_path.exists():
            test_audio_path.unlink()
        if midi_path.exists():
            midi_path.unlink()
        # Also remove .ly file if it exists
        ly_path = output_dir / "sheet_music.ly"
        if ly_path.exists():
            ly_path.unlink()
        print("✓ Intermediate files cleaned up (use --debug to keep them)")
    else:
        print(f"✓ Intermediate files kept in {output_dir}/")
        print(f"  - {test_audio_path.name}")
        print(f"  - {midi_path.name}")

    # Final summary
    print("\n" + "=" * 70)
    print("SPIKE COMPLETE")
    print("=" * 70)
    print(f"\n✓ Output: {pdf_path}")
    print("\nValidation checklist:")
    print("  [ ] Open PDF and check: Is notation readable?")
    print("  [ ] Are barlines and measures correct?")
    print("  [ ] Is treble 8vb clef visible?")
    print("  [ ] Are rhythm values appropriate?")
    print("  [ ] Does polyphonic content render correctly?")
    print("\nNext steps:")
    print("  1. If spike succeeds → proceed to Phase 2 (generalize)")
    print("  2. If fundamental blockers → reassess approach")
    print("  3. If known limitations → document and proceed")


if __name__ == "__main__":
    main()
