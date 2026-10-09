import os
import json
from src.transcriber import transcribe_audio, save_transcript
from src.hook_detector import score_and_extract_hooks

def run_pipeline(audio_path: str = None, transcript_path: str = None):
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    if transcript_path and os.path.exists(transcript_path):
        print(f"Loading existing transcript from {transcript_path}...")
        with open(transcript_path, "r") as f:
            transcript = json.load(f)
    elif audio_path and os.path.exists(audio_path):
        print(f"Transcribing audio file: {audio_path}...")
        transcript = transcribe_audio(audio_path, model_size="tiny")
        save_transcript(transcript, os.path.join(output_dir, "transcript.json"))
    else:
        # Generate mock transcript for headless verification
        print("No audio input provided. Generating sample dialogue transcript...")
        transcript = {
            "text": "Welcome to NetVibes Engine.",
            "segments": [
                {"start": 0.0, "end": 10.0, "text": "Welcome to the NetVibes Engine automated video pipeline demonstration."},
                {"start": 10.0, "end": 25.0, "text": "We extract word level dialogue and parse high performing short form candidate hooks."},
                {"start": 25.0, "end": 42.0, "text": "This clip contains high density spoken word dialogue perfect for short form vertical video content generation."},
                {"start": 42.0, "end": 65.0, "text": "The hook detector scans continuous dialogue blocks between thirty and sixty seconds ranked strictly by dialogue density and speech cadence."}
            ]
        }
        save_transcript(transcript, os.path.join(output_dir, "sample_transcript.json"))

    print("Extracting dual-stream (Dialogue + Action) hook candidates...")
    hooks = score_and_extract_hooks(transcript, video_path=audio_path or "", min_sec=30.0, max_sec=60.0)
    
    hooks_file = os.path.join(output_dir, "candidate_hooks.json")
    with open(hooks_file, "w") as f:
        json.dump(hooks, f, indent=2)
        
    print(f"Phase 1 complete! Extracted {len(hooks)} candidates -> {hooks_file}")
    return hooks

if __name__ == "__main__":
    import sys
    audio_file = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(audio_path=audio_file)
