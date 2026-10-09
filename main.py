import argparse
import os
import json
from src.transcriber import transcribe_audio, save_transcript
from src.hook_detector import score_and_extract_hooks

def main():
    parser = argparse.ArgumentParser(description="NetVibes Phase 1 - Dialogue & Hook Ingestion")
    parser.add_argument("--input", required=True, help="Path to input audio/video file")
    parser.add_argument("--output", default="output/candidate_hooks.json", help="Output path for candidate hooks JSON")
    parser.add_argument("--model", default="small.en", help="Whisper model size (default: small.en)")
    args = parser.parse_args()

    input_file = args.input
    output_hooks_file = args.output
    
    output_dir = os.path.dirname(output_hooks_file) or "output"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Transcribing audio from: {input_file} using Whisper model '{args.model}'...")
    transcript = transcribe_audio(input_file, model_size=args.model)
    
    transcript_out = os.path.join(output_dir, "transcript.json")
    save_transcript(transcript, transcript_out)
    print(f"Full transcript saved to {transcript_out}")

    print("Analyzing dialogue density and extracting top candidate hooks...")
    hooks = score_and_extract_hooks(transcript, min_sec=30.0, max_sec=60.0)
    
    with open(output_hooks_file, "w") as f:
        json.dump(hooks, f, indent=2)
        
    print(f"Extracted {len(hooks)} candidate hooks -> {output_hooks_file}")
    for idx, h in enumerate(hooks, 1):
        print(f" [{idx}] Clip: {h['clip_id']} | Range: {h['start_time']}s - {h['end_time']}s | Density: {h['density_wps']} WPS")
        print(f"     Preview: {h['preview']}")

if __name__ == "__main__":
    main()
