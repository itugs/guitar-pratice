# Guitar Practice Suite

AI-powered guitar isolation tool for practice. Extract guitar tracks from YouTube videos or local audio files using Demucs, with pitch-preserved time stretching.

## Features

- **Guitar Isolation**: Extract clean guitar tracks from songs using Demucs AI
- **Sheet Music Generation**: Automatically transcribe guitar to readable sheet music (NEW)
- **YouTube Support**: Download and process audio directly from YouTube URLs
- **Pitch-Preserved Slowdown**: Practice at 0.5x-2.0x speed without pitch shift
- **High Quality**: Uses htdemucs_6s model (state-of-the-art separation)
- **GPU Accelerated**: Automatically uses CUDA or Apple Silicon when available

## Installation

### System Requirements

**Required:**
- Python 3.8-3.11 (3.12+ not yet supported by basic-pitch)
- FFmpeg: `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Linux)

**Optional:**
- RubberBand: `brew install rubberband` (macOS) or `apt install rubberband-cli` (Linux) - required for time stretching
- LilyPond: `brew install lilypond` (macOS) or `apt install lilypond` (Linux) - required for sheet music generation
- NVIDIA GPU with CUDA (8GB+ VRAM) for faster processing
- Apple Silicon Mac (M1/M2/M3/M4) also supports GPU acceleration

### Install from source

```bash
cd guitar-practice
pip install -e .
```

## Usage

### Basic Examples

```bash
# Extract guitar from YouTube video
guitar-practice "https://www.youtube.com/watch?v=..." -o guitar.mp3

# Extract from local file
guitar-practice song.mp3 -o guitar.mp3

# Generate sheet music alongside audio
guitar-practice song.mp3 -o guitar.mp3 --sheet music.pdf

# Slow down to 75% speed (pitch preserved) with sheet music
guitar-practice song.mp3 --speed 0.75 -o slow_guitar.mp3 --sheet music.pdf

# High quality export (320kbps)
guitar-practice song.mp3 -o guitar.mp3 --bitrate 320
```

### Options

```
Options:
  -o, --output PATH       Output MP3 file path (required)
  --speed FLOAT           Playback speed (0.25-4.0, default: 1.0)
  --bitrate INT           MP3 bitrate in kbps (default: 192)
  --sheet PATH            Generate sheet music PDF at this path
  --max-duration INT      Max audio duration for sheet generation in seconds (default: 300)
  --keep-temp             Keep temporary files for debugging
  --help                  Show this message and exit
```

## Performance

Processing time depends on hardware:

- **Apple Silicon (M1/M2/M3/M4) or NVIDIA GPU**: 1-2 minutes per song
- **CPU only**: 5-10 minutes per song

First run downloads the Demucs model (~2GB), subsequent runs use cached model.

## How It Works

1. **Download/Load**: Fetches audio from YouTube or loads local file
2. **Separation**: Demucs AI splits audio into 6 stems (vocals, drums, bass, guitar, piano, other)
3. **Extraction**: Isolates the guitar stem
4. **Processing**: Applies pitch-preserved time stretching if requested
5. **Export**: Converts to high-quality MP3
6. **Transcription** (optional): Uses basic-pitch to transcribe guitar audio to MIDI
7. **Notation** (optional): Converts MIDI to professional sheet music PDF with LilyPond

## Troubleshooting

**"FFmpeg not found"**
- Install FFmpeg: `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Linux)

**"RubberBand library not found"**
- Install RubberBand: `brew install rubberband` (macOS)

**"No GPU detected - using CPU"**
- Normal on machines without CUDA or Apple Silicon
- Processing will be slower (5-10 min) but still works

**"Out of memory"**
- Close other applications
- Process shorter audio clips
- Tool will automatically fall back to CPU if GPU runs out of memory

**"LilyPond not found"** (when using --sheet)
- Install LilyPond: `brew install lilypond` (macOS) or `apt install lilypond` (Linux)
- Sheet music generation is optional - audio will still be saved if LilyPond is missing

**Sheet music quality**
- Generated sheet music is draft notation for practice and study
- May need manual cleanup in MuseScore for perfect transcription
- Works best with clear, isolated guitar parts
- Polyphonic content (chords) is supported but may require editing

## Development

Run tests:
```bash
pytest
```

## License

MIT

## Credits

Built with:
- [Demucs](https://github.com/facebookresearch/demucs) by Meta Research - AI-powered source separation
- [basic-pitch](https://github.com/spotify/basic-pitch) by Spotify - polyphonic audio-to-MIDI transcription
- [music21](https://github.com/cuthbertLab/music21) - music notation library
- [LilyPond](https://lilypond.org/) - professional music engraving
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - YouTube download
- [pyrubberband](https://github.com/bmcfee/pyrubberband) - pitch-preserved time stretching
