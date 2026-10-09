import argparse
import os
import json
import cv2
import numpy as np
from src.smart_cropper import SmartCropper
from src.renderer import render_vertical_clip

def run_phase2_pipeline(video_path: str, hooks_path: str = "output/candidate_hooks.json"):
    if not os.path.exists(hooks_path):
        print(f"Hooks file {hooks_path} not found!")
        return []
        
    with open(hooks_path, "r") as f:
        hooks = json.load(f)
        
    cropper = SmartCropper(model_name="yolov8n.pt")
    rendered_outputs = []
    output_dir = "output/rendered_clips"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process top candidates
    for idx, hook in enumerate(hooks[:3]):
        clip_id = hook["clip_id"]
        start_time = hook["start_time"]
        duration = hook["duration"]
        
        print(f"\n[Phase 2 - Processing {clip_id}] Start: {start_time}s | Duration: {duration}s")
        print(f"Calculating dynamic crop coordinates with SmartCropper...")
        crop_x, crop_w, crop_h = cropper.calculate_crop_centers(video_path, start_time, duration)
        print(f"Optimal Crop -> X: {crop_x}, Width: {crop_w}, Height: {crop_h}")
        
        out_file = os.path.join(output_dir, f"{clip_id}_vertical.mp4")
        print(f"Rendering 9:16 vertical clip to {out_file}...")
        rendered_path = render_vertical_clip(
            video_path=video_path,
            start_time=start_time,
            duration=duration,
            crop_x=crop_x,
            crop_w=crop_w,
            crop_h=crop_h,
            output_path=out_file
        )
        rendered_outputs.append({
            "clip_id": clip_id,
            "output_path": rendered_path,
            "crop_x": crop_x,
            "crop_w": crop_w,
            "crop_h": crop_h
        })
        
    print(f"\nPhase 2 Complete! Rendered {len(rendered_outputs)} clips in {output_dir}")
    return rendered_outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NetVibes Phase 2 - Smart Reframer & Video Slicer")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--hooks", default="output/candidate_hooks.json", help="Path to candidate hooks JSON")
    args = parser.parse_args()
    
    run_phase2_pipeline(video_path=args.video, hooks_path=args.hooks)
