import os

def generate_srt(segments: list, start_offset: float, end_offset: float, output_srt: str):
    """
    Filters transcript segments within [start_offset, end_offset], 
    normalizes timestamps relative to clip start (00:00:00), and writes an SRT.
    """
    os.makedirs(os.path.dirname(output_srt) or ".", exist_ok=True)

    def format_ts(seconds: float) -> str:
        millis = int((seconds % 1) * 1000)
        s = int(seconds)
        mins, secs = divmod(s, 60)
        hrs, mins = divmod(mins, 60)
        return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"

    lines = []
    idx = 1
    for seg in segments:
        s_start, s_end = seg["start"], seg["end"]
        if s_end <= start_offset or s_start >= end_offset:
            continue
        rel_start = max(0.0, s_start - start_offset)
        rel_end = min(end_offset - start_offset, s_end - start_offset)
        if rel_end <= rel_start:
            continue

        lines.extend([str(idx), f"{format_ts(rel_start)} --> {format_ts(rel_end)}", seg["text"].strip(), ""])
        idx += 1

    with open(output_srt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return output_srt
