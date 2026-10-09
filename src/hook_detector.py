import json

def score_and_extract_hooks(transcript_data: dict, min_sec: float = 30.0, max_sec: float = 60.0):
    """
    Scans segments to find continuous dialogue blocks between min_sec and max_sec 
    ranked by dialogue density (words per second).
    """
    segments = transcript_data.get("segments", [])
    candidates = []
    
    current_chunk = []
    chunk_start = 0.0
    
    for seg in segments:
        if not current_chunk:
            chunk_start = seg["start"]
        
        current_chunk.append(seg)
        chunk_duration = seg["end"] - chunk_start
        
        if min_sec <= chunk_duration <= max_sec:
            text = " ".join([s["text"].strip() for s in current_chunk])
            word_count = len(text.split())
            wps = word_count / chunk_duration if chunk_duration > 0 else 0
            
            candidates.append({
                "clip_id": f"clip_{len(candidates) + 1:02d}",
                "start_time": round(chunk_start, 2),
                "end_time": round(seg["end"], 2),
                "duration": round(chunk_duration, 2),
                "word_count": word_count,
                "density_wps": round(wps, 2),
                "preview": text[:140] + ("..." if len(text) > 140 else "")
            })
            # Slide the window forward
            current_chunk = current_chunk[1:]
            if current_chunk:
                chunk_start = current_chunk[0]["start"]

    # Sort candidates by speech density
    candidates.sort(key=lambda x: x["density_wps"], reverse=True)
    return candidates[:5]
