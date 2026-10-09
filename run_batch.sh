#!/usr/bin/env bash
set -e

# NetVibes Engine - Batch Video Processor
# Takes an input folder of full videos, loops through Phase 1, Phase 2, and Phase 3 automatically,
# and outputs ready-to-post 9:16 vertical shorts to output/final_shorts/

INPUT_DIR="${1:-input}"
CTA_TEXT="${2:-Stream on Prime Video | amzn.to/TomorrowWar}"
OUTPUT_FINAL="output/final_shorts"

echo "=================================================="
echo "   NetVibes Engine - Automated Batch Pipeline     "
echo "=================================================="
echo "Input Folder : ${INPUT_DIR}"
echo "CTA Banner   : ${CTA_TEXT}"
echo "Final Output : ${OUTPUT_FINAL}"
echo "=================================================="

if [ ! -d "$INPUT_DIR" ]; then
    echo "[Error] Input directory '$INPUT_DIR' does not exist."
    exit 1
fi

mkdir -p "$OUTPUT_FINAL"

# Activate venv if available
if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Find all matching video files
shopt -s nullglob
VIDEOS=("$INPUT_DIR"/*.mp4 "$INPUT_DIR"/*.mkv "$INPUT_DIR"/*.mov "$INPUT_DIR"/*.avi)

if [ ${#VIDEOS[@]} -eq 0 ]; then
    echo "[!] No video files (.mp4, .mkv, .mov, .avi) found in '$INPUT_DIR'."
    exit 0
fi

echo "Found ${#VIDEOS[@]} video(s) to process."

for video in "${VIDEOS[@]}"; do
    echo ""
    echo "--------------------------------------------------"
    echo "Processing Video: $video"
    echo "--------------------------------------------------"
    
    # Phase 1: Transcribe & Extract Hook Candidates
    echo "[Phase 1] Extracting transcript and hook candidates..."
    python3 run_phase1.py "$video"
    
    # Phase 2: Smart Cropping & vertical 9:16 rendering
    echo "[Phase 2] Reframing and rendering 9:16 vertical clips..."
    python3 run_phase2.py --video "$video" --hooks output/candidate_hooks.json
    
    # Phase 3: Dynamic subtitles & Lower-third CTA branding
    echo "[Phase 3] Generating subtitles and burning CTA banner..."
    python3 run_phase3.py --cta "$CTA_TEXT"
    
    echo "[✓] Finished processing: $video"
done

echo ""
echo "=================================================="
echo "[SUCCESS] Batch processing complete!"
echo "Final shorts saved in: $OUTPUT_FINAL"
echo "=================================================="
