import re
import math
import subprocess
import os
import json

def extract_action_metrics(video_path: str) -> tuple[list[float], dict[str, float]]:
    """
    Extracts scene change timestamps and audio volume levels per second from video using FFmpeg.
    """
    if not video_path or not os.path.exists(video_path):
        return [], {}

    # 1. Detect Scene Change Timestamps
    scene_cuts = []
    cmd_scene = [
        "ffmpeg", "-i", video_path,
        "-filter_complex", "select='gt(scene,0.25)',metadata=print",
        "-f", "null", "-"
    ]
    try:
        res = subprocess.run(cmd_scene, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
        matches = re.findall(r"pts_time:([\d\.]+)", res.stderr)
        scene_cuts = [float(m) for m in matches]
    except Exception as e:
        print(f"[hook_detector] Scene detection skipped/failed: {e}")

    # 2. Extract Audio Volume/Energy
    vol_cmd = [
        "ffmpeg", "-i", video_path,
        "-af", "volumedetect",
        "-f", "null", "-"
    ]
    max_vol = -20.0
    try:
        res_vol = subprocess.run(vol_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
        mean_match = re.search(r"mean_volume:\s*([-\d\.]+)\s*dB", res_vol.stderr)
        if mean_match:
            max_vol = float(mean_match.group(1))
    except Exception:
        pass

    return scene_cuts, {"mean_volume": max_vol}

def is_overlapping(cand_a: dict, cand_b: dict, min_buffer: float = 0.0) -> bool:
    a_start, a_end = cand_a["start_time"], cand_a["end_time"]
    b_start, b_end = cand_b["start_time"], cand_b["end_time"]
    return not (a_end + min_buffer <= b_start or b_end + min_buffer <= a_start)

def score_and_extract_hooks(
    transcript_data: dict, 
    video_path: str = "", 
    min_sec: float = 30.0, 
    max_sec: float = 60.0,
    target_count: int = 6,
    min_buffer: float = 0.0
) -> list[dict]:
    """
    Dual-Stream (Dialogue Drama + High-Octane Action) Hook Candidate Selection Engine.
    Enforces quota allocation and minimum non-overlapping time buffers between clips.
    """
    segments = transcript_data.get("segments", [])
    
    # Extract Action Metrics if video provided
    scene_cuts, audio_info = extract_action_metrics(video_path) if video_path else ([], {})

    all_windows = []
    
    # 1. Generate Candidate Windows using sliding step
    # Determine video total duration
    if video_path and os.path.exists(video_path):
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if fps > 0 and frame_count > 0:
                total_dur = frame_count / fps
            else:
                total_dur = segments[-1]["end"] if segments else 180.0
            cap.release()
        except Exception:
            total_dur = segments[-1]["end"] if segments else 180.0
    else:
        total_dur = segments[-1]["end"] if segments else 180.0
    step = 5.0
    
    curr_time = 0.0
    while curr_time + min_sec <= total_dur:
        win_start = curr_time
        win_end = min(total_dur, curr_time + min_sec)
        win_dur = win_end - win_start
        
        # Dialogue Density (Words Per Second)
        win_segs = [s for s in segments if not (s["end"] <= win_start or s["start"] >= win_end)]
        text = " ".join([s["text"].strip() for s in win_segs])
        words = len(text.split())
        wps = words / win_dur if win_dur > 0 else 0.0
        
        # Action Score (Scene cuts in window + audio energy density)
        cuts_in_win = len([t for t in scene_cuts if win_start <= t <= win_end])
        cuts_per_min = (cuts_in_win / win_dur) * 60.0 if win_dur > 0 else 0.0
        
        # Composite Action Score
        action_score = (cuts_per_min * 1.5) + (1.0 / (wps + 0.1)) # Action thrives on cuts + fast visual pace
        
        all_windows.append({
            "start_time": round(win_start, 2),
            "end_time": round(win_end, 2),
            "duration": round(win_dur, 2),
            "word_count": words,
            "density_wps": round(wps, 2),
            "scene_cuts": cuts_in_win,
            "cuts_per_min": round(cuts_per_min, 2),
            "action_score": round(action_score, 2),
            "preview": text[:140] + ("..." if len(text) > 140 else "") if text else "[High-Octane Visual / Action Sequence]"
        })
        curr_time += step

    # 2. Rank Candidates into Stream A (Dialogue) & Stream B (Action)
    dialogue_ranked = sorted(all_windows, key=lambda x: x["density_wps"], reverse=True)
    action_ranked = sorted(all_windows, key=lambda x: x["action_score"], reverse=True)

    selected = []
    target_dialogue = target_count // 2
    target_action = target_count - target_dialogue

    selected_action = []
    selected_dialogue = []

    # Select Action Candidates
    for cand in action_ranked:
        if len(selected_action) >= target_action:
            break
        if not any(is_overlapping(cand, s, min_buffer) for s in selected_action):
            cand_copy = dict(cand)
            cand_copy["stream_type"] = "Action"
            selected_action.append(cand_copy)

    # Select Dialogue Candidates
    for cand in dialogue_ranked:
        if len(selected_dialogue) >= target_dialogue:
            break
        if not any(is_overlapping(cand, s, min_buffer) for s in (selected_action + selected_dialogue)):
            cand_copy = dict(cand)
            cand_copy["stream_type"] = "Dialogue"
            selected_dialogue.append(cand_copy)

    # 3. Interleave Selections: [Action 1, Dialogue 1, Action 2, Dialogue 2...]
    final_candidates = []
    max_len = max(len(selected_action), len(selected_dialogue))
    for i in range(max_len):
        if i < len(selected_action):
            final_candidates.append(selected_action[i])
        if i < len(selected_dialogue):
            final_candidates.append(selected_dialogue[i])

    # Assign final clip_ids
    for idx, cand in enumerate(final_candidates, 1):
        cand["clip_id"] = f"clip_{idx:02d}"

    return final_candidates
