#!/bin/bash
# Quick activation script for guitar-practice venv
# Usage: source activate.sh

echo "Activating guitar-practice virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment active"
echo ""
echo "Available commands:"
echo "  guitar-practice --help"
echo "  guitar-practice song.mp3 -o guitar.mp3"
echo ""
echo "System requirements:"
echo "  - FFmpeg: brew install ffmpeg"
echo "  - RubberBand: brew install rubberband"
