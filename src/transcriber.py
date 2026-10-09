import whisper
import json
import subprocess
import os

def extract_filtered_audio(video_audio_path: str, output_wav: str = "/tmp/filtered_speech.wav") -> str:
    """
    Extracts audio with speech bandpass filtering (highpass 200Hz, lowpass 3000Hz)
    to isolate vocal frequencies and reduce background music/hallucinations.
    """
    os.makedirs(os.path.dirname(output_wav) or ".", exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", video_audio_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-af", "highpass=f=200,lowpass=f=3000",
        output_wav
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError:
        # Fallback for silent / video-only inputs
        cmd_silent = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
            "-t", "1",
            output_wav
        ]
        subprocess.run(cmd_silent, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return output_wav

def transcribe_audio(audio_path: str, model_size: str = "small.en") -> dict:
    """
    Transcribes audio with speech isolation and fp16=False CPU execution.
    """
    filtered_wav = extract_filtered_audio(audio_path)
    print(f"Loading Whisper model '{model_size}'...")
    model = whisper.load_model(model_size)
    print("Transcribing filtered vocal track with word-level timestamps...")
    result = model.transcribe(filtered_wav, word_timestamps=True, fp16=False)
    
    # Cleanup temp wav
    if os.path.exists(filtered_wav):
        try:
            os.remove(filtered_wav)
        except Exception:
            pass
            
    return result

def save_transcript(result: dict, output_json: str = "transcript.json"):
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
