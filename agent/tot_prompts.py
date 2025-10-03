# agent/tot_prompts.py
import os, time
from typing import Any, List
from tot_parse import parse_thoughts_from_text

# ToT prompt with drawing actions compatible with apply_thought
PROPOSE_TEMPLATE = """
You are solving the problem with a Tree-of-Thoughts strategy. Return ONLY a STRICT JSON array, no prose.
Each element is a thought object with keys:
{{
  "goal": "short goal text",
  "rationale": "1-3 sentences of reasoning; when done include 'FINAL_ANSWER: ...'",
  "actions": [
     {{"type": "line", "args": {{"p1": [x,y], "p2": [x,y], "color": "black", "lw": 2}}}},
     {{"type": "segment", "args": {{"p1": [x,y], "p2": [x,y]}}}},
     {{"type": "ray", "args": {{"p1": [x,y], "p2": [x,y]}}}},
     {{"type": "polyline", "args": {{"pts": [[x,y],[x,y],...]}}}},
     {{"type": "polygon", "args": {{"pts": [[x,y],[x,y],...], "edgecolor": "black", "facecolor": null}}}},
     {{"type": "point", "args": {{"p": [x,y], "label": "A"}}}},
     {{"type": "text", "args": {{"p": [x,y], "text": "label"}}}},
     {{"type": "circle", "args": {{"center": [x,y], "r": 0.2}}}},
     {{"type": "circle3", "args": {{"a": [x,y], "b": [x,y], "c": [x,y]}}}},
     {{"type": "arc", "args": {{"center": [x,y], "r": 0.2, "start": 0, "end": 90}}}},
     {{"type": "angle_mark", "args": {{"a": [x,y], "o": [x,y], "b": [x,y]}}}},
     {{"type": "perp_through", "args": {{"p": [x,y], "a1": [x,y], "a2": [x,y]}}}},
     {{"type": "parallel_through", "args": {{"p": [x,y], "a1": [x,y], "a2": [x,y]}}}}
  ],
  "expected_observation": "what you expect to see after actions",
  "stop_signal": false
}}
Rules:
- Use only the action types and arg shapes listed above.
- Coordinates are normalized in [0,1]. Keep diagrams readable.
- When you reach the final answer, set "stop_signal" to true and include 'FINAL_ANSWER: ...' in rationale.
- Do not include markdown, comments, or extra keys. Output the JSON array only.

Question: {question}
Current rationale: {rationale}
"""


def _trace_dir_from_canvas(canvas_path: str) -> str:
    base = os.path.dirname(canvas_path) if isinstance(canvas_path, str) else os.getcwd()
    d = os.path.join(base, "tot_trace")
    os.makedirs(d, exist_ok=True)
    return d


def propose_thoughts(state: Any, K: int, llm_complete) -> List[dict]:
    prompt = PROPOSE_TEMPLATE.format(
        question=getattr(state, "question", ""),
        rationale=getattr(state, "text_trace", ""),
    )

    raw = llm_complete(prompt)
    snippet = raw[:200] + "..." if isinstance(raw, str) and len(raw) > 200 else raw
    print("[ToT] propose_thoughts raw response:", snippet)

    try:
        thoughts = parse_thoughts_from_text(raw)
    except Exception as e:
        import traceback
        traceback.print_exc()
        thoughts = []
    print("[ToT] parsed thoughts count:", len(thoughts) if isinstance(thoughts, list) else "invalid")
    if isinstance(thoughts, list) and thoughts:
        # ensure dict form by wrapping strings
        normalized = []
        for t in thoughts[:K]:
            if isinstance(t, dict):
                # enforce that stop_signal=true requires at least one action
                if bool(t.get("stop_signal")) and not t.get("actions"):
                    t = {**t, "stop_signal": False}
                normalized.append(t)
            elif isinstance(t, str):
                normalized.append({
                    "goal": t,
                    "rationale": t,
                    "actions": [],
                    "expected_observation": "",
                    "stop_signal": False,
                })
        if normalized:
            return normalized

    # Log raw on failure for debugging
    try:
        trace_dir = _trace_dir_from_canvas(getattr(state, "canvas", ""))
        ts = int(time.time() * 1000)
        with open(os.path.join(trace_dir, f"propose_raw_{ts}.txt"), "w", encoding="utf-8") as f:
            f.write(raw if isinstance(raw, str) else str(raw))
    except Exception:
        pass

    # safe default (draw nothing, continue reasoning)
    return [{
        "goal": "Outline a plan",
        "rationale": "Summarize givens, define variables, and propose next steps.",
        "actions": [],
        "expected_observation": "A clearer picture of the next subgoal",
        "stop_signal": False
    }][:K]

