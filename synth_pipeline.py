import asyncio
import argparse
import json
import os
import random
import re
import shutil
import subprocess
import time
import uuid
import edge_tts

from google import genai
from google.genai import types

def get_gemini_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key and os.path.exists(os.path.expanduser("~/.bashrc")):
        try:
            with open(os.path.expanduser("~/.bashrc")) as f:
                m = re.search(r'export\s+GEMINI_API_KEY=[\"\']?([^\"\'\n]+)', f.read())
                if m:
                    key = m.group(1).strip("\"'")
        except Exception:
            pass
    if not key:
        raise RuntimeError("GEMINI_API_KEY not found in environment or ~/.bashrc.")
    return key

NATGEO_PREFIX = "National Geographic archival ocean exploration footage, abyssal zone, 35mm lens, realistic marine snow, bioluminescence, hyper-detailed textures, murky deep ocean depth-of-field, cinematic lighting, no cartoon, no 3D render look, "

def generate_scenes_from_llm(topic: str, api_key: str = None) -> list:
    key = api_key or get_gemini_api_key()
    print(f"[NetVibes LLM] Querying Gemini API via google-genai SDK for topic: '{topic}'...")
    
    client = genai.Client(api_key=key)
    
    prompt_text = (
        f"You are a short-form video scriptwriter. Create a 5-scene short script for the topic: '{topic}'.\n"
        "Requirements:\n"
        "1. Exactly 5 scenes.\n"
        "2. Each scene narration MUST be punchy and under 10 words per beat.\n"
        f"3. Each visual prompt MUST describe a cinematic 9:16 vertical image and begin with '{NATGEO_PREFIX}'\n"
        "4. Return ONLY raw JSON array of objects with keys 'narration' and 'prompt'. No markdown formatting.\n"
    )
    
    # Official google-genai SDK call for text generation
    res = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt_text
    )
    
    text_out = res.text
    clean_json = re.sub(r"```json|```", "", text_out).strip()
    scenes = json.loads(clean_json)
    if isinstance(scenes, list) and len(scenes) >= 1:
        print(f"[NetVibes LLM] Successfully generated {len(scenes)} scenes via Gemini API!")
        for s in scenes:
            if not s["prompt"].startswith("National Geographic"):
                s["prompt"] = f"{NATGEO_PREFIX}{s['prompt']}"
        return scenes[:5]
        
    raise RuntimeError(f"Failed to parse structured script from LLM output: '{text_out}'")

def generate_native_imagen_image(prompt: str, image_path: str, api_key: str = None) -> str:
    key = api_key or get_gemini_api_key()
    print(f"[NetVibes Imagen] Initializing google-genai Client...")
    client = genai.Client(api_key=key)
    
    # 1. Primary request: Imagen 3 ('imagen-3.0-generate-002') via generate_images
    try:
        print(f"[NetVibes Imagen] Attempting Imagen 3 ('imagen-3.0-generate-002') in native 9:16 aspect ratio...")
        res = client.models.generate_images(
            model="imagen-3.0-generate-002",
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                output_mime_type="image/jpeg",
                aspect_ratio="9:16",
            )
        )
        if res and res.generated_images:
            for img in res.generated_images:
                with open(image_path, "wb") as f:
                    f.write(img.image.image_bytes)
                print(f"[NetVibes Imagen] Native 9:16 image saved via imagen-3.0-generate-002 -> {image_path}")
                return image_path
    except Exception as e:
        print(f"[NetVibes Imagen] imagen-3.0-generate-002 generate_images endpoint returned: {e}")

    # 2. Native Google GenAI Image Models via generate_content (as recommended by google-genai SDK deprecation warning)
    image_models = ["gemini-3.1-flash-image", "gemini-2.5-flash-image", "gemini-3-pro-image"]
    last_err = None

    for mname in image_models:
        try:
            print(f"[NetVibes Image] Generating native 9:16 visual via '{mname}'...")
            res = client.models.generate_content(
                model=mname,
                contents=f"Generate a native 9:16 vertical aspect ratio photorealistic cinematic image: {prompt}"
            )
            if res and res.candidates:
                for candidate in res.candidates:
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if part.inline_data and part.inline_data.data:
                                with open(image_path, "wb") as f:
                                    f.write(part.inline_data.data)
                                print(f"[NetVibes Image] Native 9:16 visual saved via {mname} -> {image_path}")
                                return image_path
        except Exception as e:
            last_err = e
            print(f"[NetVibes Image] {mname} failed: {e}")

    raise RuntimeError(f"Native 9:16 Image generation failed for prompt: '{prompt}'. Error: {last_err}")

def format_ass_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs >= 100:
        cs = 99
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def get_audio_duration(audio_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return float(res.stdout.strip())

async def generate_scene_speech(text: str, voice: str, audio_path: str, offset_sec: float):
    comm = edge_tts.Communicate(text, voice, rate="-6%", boundary="WordBoundary")
    words = []
    
    with open(audio_path, "wb") as f:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                start_sec = (chunk["offset"] / 10_000_000.0) + offset_sec
                dur_sec = chunk["duration"] / 10_000_000.0
                words.append({
                    "text": chunk["text"],
                    "start": start_sec,
                    "end": start_sec + dur_sec,
                    "dur_cs": max(1, int(round(dur_sec * 100)))
                })
    return words

def chunk_words_strictly(words: list, max_words: int = 3, max_pause: float = 0.3) -> list:
    lines = []
    curr = []
    
    for w in words:
        txt = w["text"].strip()
        
        if curr:
            prev_w = curr[-1]
            prev_txt = prev_w["text"].strip()
            
            # Check 1: Hard punctuation boundary on previous word
            prev_has_punct = any(prev_txt.endswith(p) for p in [".", "!", "?", ";", ":", ",", "—", "-"])
            
            # Check 2: Silence gap / pause > 0.3s between words
            pause_gap = w["start"] - prev_w["end"]
            
            # Check 3: Max card length (2-3 words) reached
            card_full = len(curr) >= max_words
            
            # Check 4: Different scene boundary
            different_scene = prev_w.get("scene_end") != w.get("scene_end")
            
            if prev_has_punct or pause_gap > max_pause or card_full or different_scene:
                lines.append(curr)
                curr = []
                
        curr.append(w)
        
        # Flush card immediately if current word ends with punctuation
        if any(txt.endswith(p) for p in [".", "!", "?", ";", ":", ",", "—", "-"]):
            lines.append(curr)
            curr = []
            
    if curr:
        lines.append(curr)
        
    return lines

def build_karaoke_ass(words: list, ass_path: str, words_per_line: int = 3):
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke, DejaVu Sans, 80, &H0000FFFF, &H00FFFFFF, &H00000000, &H80000000, 1, 0, 0, 0, 100, 100, 0, 0, 1, 5, 2, 2, 50, 50, 280

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    lines = chunk_words_strictly(words, max_words=words_per_line, max_pause=0.3)
    
    for idx, line_words in enumerate(lines):
        if not line_words:
            continue
        line_start_sec = line_words[0]["start"]
        
        # Clip strictly to speech boundary: exact end timestamp of final word in phrase
        line_end_sec = line_words[-1]["end"]
        
        # HARD CLEAR ON SCENE CUTS: Never exceed the scene boundary timestamp!
        scene_end_cap = line_words[-1].get("scene_end")
        if scene_end_cap:
            line_end_sec = min(line_end_sec, scene_end_cap)
            
        if idx + 1 < len(lines) and lines[idx + 1]:
            next_start_sec = lines[idx + 1][0]["start"]
            line_end_sec = min(line_end_sec, next_start_sec - 0.05)
            
        if line_end_sec <= line_start_sec:
            line_end_sec = line_start_sec + 0.1
            
        line_start = format_ass_time(line_start_sec)
        line_end = format_ass_time(line_end_sec)
        
        line_str = ""
        for w in line_words:
            line_str += f"{{\\kf{w['dur_cs']}}}{w['text'].upper()} "
        
        events.append(f"Dialogue: 0,{line_start},{line_end},Karaoke,,0,0,0,,{line_str.strip()}")
    
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events) + "\n")

def render_scene_kenburns(image_path: str, duration: float, output_video_path: str, motion_type: str = "zoom_in"):
    clip_frames = max(30, int(round(duration * 30)))
    
    if motion_type == "zoom_in":
        zoom_expr = f"1.0+0.08*(on/{clip_frames})"
    elif motion_type == "zoom_out":
        zoom_expr = f"1.08-0.08*(on/{clip_frames})"
    else:
        zoom_expr = f"1.0+0.08*(on/{clip_frames})"
        
    x_expr = "iw/2-(iw/zoom/2)"
    y_expr = "ih/2-(ih/zoom/2)"
        
    filter_graph = f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}':d={clip_frames}:s=1080x1920:fps=30"
    
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", image_path,
        "-vf", filter_graph,
        "-vframes", str(clip_frames),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30",
        output_video_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

async def process_multiscene_pipeline(scenes: list, voice: str, output_path: str, api_key: str = None):
    tmp_dir = "/tmp/synth_multiscene"
    
    print(f"[NetVibes Workspace] Purging stale workspace in '{tmp_dir}'...")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    global_words = []
    global_time_offset = 0.0
    scene_cut_times = [0.0]
    video_clips = []
    audio_clips = []
    
    motions = ["zoom_in", "zoom_out"]
    
    print(f"[NetVibes Synth] Starting Multi-Scene Pipeline ({len(scenes)} scenes)...")
    
    for idx, scene in enumerate(scenes):
        print(f"\n--- Processing Scene {idx+1}/{len(scenes)} ---")
        print(f"Narration: \"{scene['narration']}\"")
        print(f"Visual Prompt: \"{scene['prompt']}\"")
        
        unique_id = uuid.uuid4().hex[:6]
        scene_audio = os.path.join(tmp_dir, f"audio_scene_{idx}_{unique_id}.mp3")
        scene_img = os.path.join(tmp_dir, f"img_scene_{idx}_{unique_id}.jpg")
        scene_video = os.path.join(tmp_dir, f"video_scene_{idx}_{unique_id}.mp4")
        
        # 1. Audio & Speech
        words = await generate_scene_speech(scene["narration"], voice, scene_audio, global_time_offset)
        duration = get_audio_duration(scene_audio)
        scene_end_time = global_time_offset + duration
        for w in words:
            w["scene_end"] = scene_end_time
        global_words.extend(words)
        
        print(f"Scene {idx+1} audio duration: {duration:.2f}s, words: {len(words)}")
        
        # 2. Render scene via Imagen 3 (imagen-3.0-generate-002) with 9:16 aspect ratio
        generate_native_imagen_image(scene["prompt"], scene_img, api_key=api_key)
        
        # 3. Ken Burns Motion Clip
        motion = motions[idx % len(motions)]
        clip_frames = max(30, int(round(duration * 30)))
        print(f"Rendering Ken Burns motion video ({motion}, duration: {duration:.2f}s, frames: {clip_frames})...")
        render_scene_kenburns(scene_img, duration, scene_video, motion_type=motion)
        
        video_clips.append(scene_video)
        audio_clips.append(scene_audio)
        global_time_offset += duration
        scene_cut_times.append(global_time_offset)

    print(f"\n[NetVibes Synth] Total video length: {global_time_offset:.2f}s across {len(scenes)} scenes.")
    
    # 4. Concatenate Video Clips
    video_concat_txt = os.path.join(tmp_dir, "video_concat.txt")
    with open(video_concat_txt, "w") as f:
        for v in video_clips:
            f.write(f"file '{v}'\n")
            
    raw_video_concat = os.path.join(tmp_dir, "concat_video.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", video_concat_txt,
        "-c", "copy", raw_video_concat
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    
    # 5. Concatenate Audio Clips
    audio_concat_txt = os.path.join(tmp_dir, "audio_concat.txt")
    with open(audio_concat_txt, "w") as f:
        for a in audio_clips:
            f.write(f"file '{a}'\n")
            
    raw_audio_concat = os.path.join(tmp_dir, "concat_audio.mp3")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", audio_concat_txt,
        "-c", "copy", raw_audio_concat
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    
    # 6. Global ASS Karaoke Subtitles, Ambient Drone & Sonar Pulse SFX
    ass_path = os.path.join(tmp_dir, "global_subs.ass")
    build_karaoke_ass(global_words, ass_path, words_per_line=3)
    
    ambient_drone = os.path.join(tmp_dir, "ambient_drone.wav")
    print(f"[NetVibes Audio] Synthesizing low-frequency ambient drone/sub-creak layer ({global_time_offset:.2f}s) at -18dB...")
    drone_cmd = [
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"aevalsrc=0.5*sin(2*PI*40*t)+0.3*sin(2*PI*80*t)+0.15*sin(2*PI*120*t):s=24000:d={global_time_offset:.2f}",
        "-af", "lowpass=f=180,volume=-18dB",
        ambient_drone
    ]
    subprocess.run(drone_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    sonar_pulse_wav = os.path.join(tmp_dir, "sonar_pulse.wav")
    print(f"[NetVibes SFX] Generating sonar pulse/impact accent for scene cuts...")
    pulse_cmd = [
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "aevalsrc=0.7*sin(2*PI*(750*exp(-10*t)+50))*exp(-5*t)+0.5*sin(2*PI*55*exp(-3*t))*exp(-3*t):s=24000:d=0.8",
        "-af", "volume=-10dB",
        sonar_pulse_wav
    ]
    subprocess.run(pulse_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    # Build sfx_track across scene cut timestamps
    sfx_track_wav = os.path.join(tmp_dir, "sfx_track.wav")
    valid_cuts = scene_cut_times[:-1]
    inputs = ["-i", sonar_pulse_wav] * len(valid_cuts)
    filter_parts = []
    for idx_c, t_cut in enumerate(valid_cuts):
        ms = int(t_cut * 1000)
        filter_parts.append(f"[{idx_c}:a]adelay={ms}|{ms}[sfx{idx_c}]")
    mix_inputs = "".join([f"[sfx{i}]" for i in range(len(valid_cuts))])
    filter_str = ";".join(filter_parts) + f";{mix_inputs}amix=inputs={len(valid_cuts)}:duration=first[aout]"
    cmd_sfx = ["ffmpeg", "-y"] + inputs + ["-filter_complex", filter_str, "-map", "[aout]", sfx_track_wav]
    subprocess.run(cmd_sfx, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    
    # 7. Final Merge & Subtitle Burn with Audio Atmosphere Mixing
    escaped_ass = ass_path.replace(":", "\\:").replace("'", "\\'")
    
    final_cmd = [
        "ffmpeg", "-y",
        "-i", raw_video_concat,
        "-i", raw_audio_concat,
        "-i", ambient_drone,
        "-i", sfx_track_wav,
        "-filter_complex", f"[1:a]volume=1.0[voc];[2:a]volume=1.0[bg];[3:a]volume=1.0[sfx];[voc][bg][sfx]amix=inputs=3:duration=first[aout];[0:v]ass='{escaped_ass}'[vout]",
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        output_path
    ]
    
    print("\n[NetVibes Synth] Multiplexing multi-scene video, audio & ASS subtitles...")
    subprocess.run(final_cmd, check=True)
    print(f"\n[NetVibes Synth] Multi-Scene Synthetic Short Complete -> {output_path}")

async def main():
    parser = argparse.ArgumentParser(description="NetVibes Multi-Scene Native Imagen 3 Pipeline")
    parser.add_argument("--topic", type=str, default="The Dark Reality of Deep Sea Trenches", help="Topic string to generate scenes via LLM")
    parser.add_argument("--scenes_json", type=str, default="", help="Path to JSON file containing list of scenes")
    parser.add_argument("--voice", type=str, default="en-US-ChristopherNeural", help="Edge TTS voice")
    parser.add_argument("--gemini_key", type=str, default="", help="Gemini API key for LLM and Imagen generation")
    parser.add_argument("--output", type=str, default="/home/yayock79/netvibes-engine/output/synth_shorts/synth_clip_multiscene.mp4", help="Output MP4 file path")
    args = parser.parse_args()

    api_key = args.gemini_key or get_gemini_api_key()

    if args.scenes_json and os.path.exists(args.scenes_json):
        with open(args.scenes_json, "r") as f:
            scenes = json.load(f)
    else:
        scenes = generate_scenes_from_llm(args.topic, api_key=api_key)

    await process_multiscene_pipeline(scenes, args.voice, args.output, api_key=api_key)

if __name__ == "__main__":
    asyncio.run(main())
