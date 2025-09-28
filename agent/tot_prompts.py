# agent/tot_prompts.py
import os, io, base64, time
from typing import Any, List
from PIL import Image
from tot_parse import parse_thoughts_from_text   # if you’re running scripts directly, use: from tot_parse import ...

PROPOSE_TEMPLATE = """You are a VisualSketchpad Tree-of-Thoughts planner.
Return a STRICT JSON array (double quotes, no trailing commas, no comments).
Each element uses this schema:
{{
  "goal": "str",
  "actions": [{{"type":"str", "args":{{}}}}],
  "expected_observation": "str",
  "stop_signal": false
}}
ONLY output the JSON array. No prose, no explanations.
Question: {question}
Current rationale: {rationale}
SketchpadThumbnail(Base64PNG): {thumb}
"""

def _thumbnail_b64(pil_img: Image.Image, max_wh: int = 256) -> str:
    pil_img = pil_img.copy()
    pil_img.thumbnail((max_wh, max_wh))
    buf = io.BytesIO(); pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _ensure_image(canvas: Any) -> Image.Image:
    if isinstance(canvas, Image.Image):
        return canvas
    if isinstance(canvas, (bytes, bytearray)):
        return Image.open(io.BytesIO(canvas)).convert("RGB")
    if isinstance(canvas, str) and os.path.exists(canvas):
        return Image.open(canvas).convert("RGB")
    return Image.new("RGB", (512, 512), "white")

def _trace_dir_from_canvas(canvas_path: str) -> str:
    # canvas is .../<instance>/tot_canvas.png
    base = os.path.dirname(canvas_path) if isinstance(canvas_path, str) else os.getcwd()
    d = os.path.join(base, "tot_trace")
    os.makedirs(d, exist_ok=True)
    return d

def propose_thoughts(state: Any, K: int, llm_complete) -> List[dict]:
    img = _ensure_image(getattr(state, "canvas", None))
    prompt = PROPOSE_TEMPLATE.format(
        question=getattr(state, "question", ""),
        rationale=getattr(state, "text_trace", ""),
        thumb=_thumbnail_b64(img)
    )

    raw = llm_complete(prompt)

    # try to parse
    thoughts = parse_thoughts_from_text(raw)
    if thoughts:
        return thoughts[:K]

    # Log raw on failure for debugging
    try:
        trace_dir = _trace_dir_from_canvas(getattr(state, "canvas", ""))
        ts = int(time.time() * 1000)
        with open(os.path.join(trace_dir, f"propose_raw_{ts}.txt"), "w", encoding="utf-8") as f:
            f.write(raw if isinstance(raw, str) else str(raw))
    except Exception:
        pass

    # safe default
    return [{
        "goal": "Initial scene read and plan",
        "actions": [{"type":"read_scene", "args":{"what":"objects, relations, axes, key points"}}],
        "expected_observation": "List of salient entities and candidate regions",
        "stop_signal": False
    }][:K]
