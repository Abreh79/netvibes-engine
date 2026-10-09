import argparse
import os
import json
from src.caption_generator import generate_srt
from src.renderer import render_final_attributed_clip

def run_phase3_pipeline(cta_text: str = "Stream on Prime Video | amzn.to/TomorrowWar"):
    output_dir = "output"
    transcript_file = os.path.join(output_dir, "transcript.json")
    if not os.path.exists(transcript_file):
        transcript_file = os.path.join(output_dir, "sample_transcript.json")
        
    hooks_file = os.path.join(output_dir, "candidate_hooks.json")
    
    if not os.path.exists(hooks_file) or not os.path.exists(transcript_file):
        print("Required input files missing. Run Phase 1 & Phase 2 first!")
        return []
        
    with open(transcript_file, "r") as f:
        transcript = json.load(f)
    segments = transcript.get("segments", [])
    
    with open(hooks_file, "r") as f:
        hooks = json.load(f)
        
    sub_dir = os.path.join(output_dir, "subtitles")
    final_dir = os.path.join(output_dir, "final_shorts")
    os.makedirs(sub_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    
    final_results = []
    
    for hook in hooks[:3]:
        clip_id = hook["clip_id"]
        start_time = hook["start_time"]
        end_time = hook["end_time"]
        
        vertical_mp4 = os.path.join(output_dir, "rendered_clips", f"{clip_id}_vertical.mp4")
        if not os.path.exists(vertical_mp4):
            print(f"Vertical clip {vertical_mp4} not found, skipping...")
            continue
            
        srt_file = os.path.join(sub_dir, f"{clip_id}.srt")
        print(f"\n[Phase 3 - Subtitles] {clip_id} ({start_time}s -> {end_time}s)")
        generate_srt(segments, start_time, end_time, srt_file)
        
        out_final_mp4 = os.path.join(final_dir, f"{clip_id}_final_attributed.mp4")
        print(f"Burning captions & CTA banner '{cta_text}' -> {out_final_mp4}...")
        render_final_attributed_clip(
            input_vertical_mp4=vertical_mp4,
            srt_path=srt_file,
            cta_text=cta_text,
            output_path=out_final_mp4
        )
        
        final_results.append({
            "clip_id": clip_id,
            "srt_path": srt_file,
            "final_video": out_final_mp4,
            "preview": hook.get("preview", ""),
            "start_time": start_time,
            "end_time": end_time,
            "density_wps": hook.get("density_wps", 0)
        })
        
    print(f"\nPhase 3 Complete! Produced {len(final_results)} final attributed short-form videos in {final_dir}")
    return final_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NetVibes Phase 3 - Dynamic Subtitles & Attribution Banner")
    parser.add_argument("--cta", default="Stream on Prime Video | amzn.to/TomorrowWar", help="CTA Banner Text")
    args = parser.parse_args()
    
    run_phase3_pipeline(cta_text=args.cta)
