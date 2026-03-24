# Guitar Practice Suite

AI-powered guitar isolation tool for practice. Extract guitar tracks from YouTube videos or local audio files using Demucs, with pitch-preserved time stretching.

## Features

- **Guitar Isolation**: Extract clean guitar tracks from songs using Demucs AI
- **YouTube Support**: Download and process audio directly from YouTube URLs
- **Pitch-Preserved Slowdown**: Practice at 0.5x-2.0x speed without pitch shift
- **High Quality**: Uses htdemucs_6s model (state-of-the-art separation)
- **GPU Accelerated**: Automatically uses CUDA or Apple Silicon when available

## Installation

### System Requirements

**Required:**
- Python 3.8+
- FFmpeg: `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Linux)
- RubberBand: `brew install rubberband` (macOS) or `apt install rubberband-cli` (Linux)

**Optional:**
- NVIDIA GPU with CUDA (8GB+ VRAM) for faster processing
- Apple Silicon Mac (M1/M2) also supports GPU acceleration

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

# Slow down to 75% speed (pitch preserved)
guitar-practice song.mp3 --speed 0.75 -o slow_guitar.mp3

# High quality export (320kbps)
guitar-practice song.mp3 -o guitar.mp3 --bitrate 320
```

### Options

```
Options:
  -o, --output PATH    Output MP3 file path (required)
  --speed FLOAT        Playback speed (0.25-4.0, default: 1.0)
  --bitrate INT        MP3 bitrate in kbps (default: 192)
  --keep-temp          Keep temporary files for debugging
  --help               Show this message and exit
```

## Performance

Processing time depends on hardware:

- **M1/M2 Mac or NVIDIA GPU**: 1-2 minutes per song
- **CPU only**: 5-10 minutes per song

First run downloads the Demucs model (~2GB), subsequent runs use cached model.

## How It Works

1. **Download/Load**: Fetches audio from YouTube or loads local file
2. **Separation**: Demucs AI splits audio into 6 stems (vocals, drums, bass, guitar, piano, other)
3. **Extraction**: Isolates the guitar stem
4. **Processing**: Applies pitch-preserved time stretching if requested
5. **Export**: Converts to high-quality MP3

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

## Development

Run tests:
```bash
pytest
```

## License

MIT

## Credits

Built with:
- [Demucs](https://github.com/facebookresearch/demucs) by Meta Research
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for YouTube download
- [pyrubberband](https://github.com/bmcfee/pyrubberband) for pitch-preserved time stretching
