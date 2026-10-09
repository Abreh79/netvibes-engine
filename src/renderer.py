import subprocess
import os

def render_vertical_clip(video_path: str, start_time: float, duration: float, 
                         crop_x: int, crop_w: int, crop_h: int, output_path: str):
    """
    Slices and crops the video to vertical 9:16 and scales to 1080x1920.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    filter_complex = f"crop={crop_w}:{crop_h}:{crop_x}:0,scale=1080:1920:flags=lanczos"

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-t", str(duration),
        "-i", video_path,
        "-vf", filter_complex,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "22",
        "-c:a", "aac",
        "-b:a", "128k",
        output_path
    ]

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return output_path

def render_final_attributed_clip(input_vertical_mp4: str, srt_path: str, cta_text: str, output_path: str):
    """
    Burns subtitles and adds an attribution CTA banner to the vertical clip.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # Escape path for ffmpeg subtitles filter
    escaped_srt = srt_path.replace(":", "\\:").replace("'", "\\'")
    
    subtitles_filter = (
        f"subtitles='{escaped_srt}':force_style='FontSize=18,PrimaryColour=&H00FFFFFF&,"
        f"OutlineColour=&H00000000&,BorderStyle=3,Outline=2,Alignment=2,MarginV=250'"
    )
    cta_filter = (
        f"drawbox=y=ih-160:color=black@0.65:width=iw:height=100:t=fill,"
        f"drawtext=text='{cta_text}':fontcolor=white:fontsize=36:x=(w-text_w)/2:y=h-125"
    )

    combined_filter = f"{subtitles_filter},{cta_filter}"

    cmd = [
        "ffmpeg", "-y",
        "-i", input_vertical_mp4,
        "-vf", combined_filter,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-c:a", "copy",
        output_path
    ]

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return output_path
