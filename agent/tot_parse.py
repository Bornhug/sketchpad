# agent/tot_parse.py
from __future__ import annotations
import json, re, ast
from typing import Any, Dict, List

SMART = {
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "—": "-", "–": "-",
}

def _norm_quotes(s: str) -> str:
    for k, v in SMART.items():
        s = s.replace(k, v)
    return s

def _strip_fences(s: str) -> str:
    # remove ```json ... ``` or ``` ... ```
    return re.sub(r"```(?:json|JSON)?\s*([\s\S]*?)```", r"\1", s, flags=re.MULTILINE)

def _remove_trailing_commas(s: str) -> str:
    # ,] or ,}  -> ] or }
    return re.sub(r",\s*([}\]])", r"\1", s)

def _extract_first_json_array(s: str) -> str:
    m = re.search(r"\[[\s\S]*\]", s)  # greedy to last ]
    return m.group(0) if m else s

def _normalize_action(a: Dict[str, Any]) -> Dict[str, Any]:
    t = a.get("type") or a.get("name") or a.get("op")
    if not t:
        raise ValueError("action missing type/name/op")
    args = a.get("args")
    if args is None:
        args = {k: v for k, v in a.items() if k not in {"type", "name", "op"}}
    return {"type": str(t), "args": args}

def _normalize_thought(t: Dict[str, Any]) -> Dict[str, Any]:
    goal = t.get("goal") or t.get("intent") or ""
    actions = t.get("actions") or t.get("plan") or []
    if not isinstance(actions, list):
        actions = [actions]
    actions = [_normalize_action(a) if isinstance(a, dict) else {"type": str(a), "args": {}} for a in actions]
    exp = t.get("expected_observation") or t.get("expected") or ""
    stop = t.get("stop_signal", False)
    stop = bool(stop) if isinstance(stop, (bool, int)) else str(stop).strip().lower() in {"true","yes","1"}
    return {"goal": str(goal), "actions": actions, "expected_observation": str(exp), "stop_signal": stop}

def _fallback_from_bullets(raw: str) -> List[Dict[str, Any]]:
    # Parse patterns like:
    # 1) Goal: ...; Actions: draw_line(...); Expected: ...; Stop: no
    thoughts = []
    blocks = re.split(r"\n\s*\d+[\).\]]\s*", "\n" + raw.strip())
    for b in blocks:
        if not b.strip(): continue
        goal = re.search(r"(?i)goal\s*:\s*(.*)", b)
        exp  = re.search(r"(?i)(expected|expected_observation)\s*:\s*(.*)", b)
        stop = re.search(r"(?i)stop(_signal)?\s*:\s*(.*)", b)
        acts = []
        # crude action lines like: - draw_line x1=.. y1=.. x2=.. y2=..
        for line in re.findall(r"(?:^|\n)\s*[-*]\s*(.*)", b):
            line = line.strip()
            if not line: continue
            m = re.match(r"([a-zA-Z_][\w]*)\s*(.*)", line)
            if m:
                typ, rest = m.group(1), m.group(2)
                args = dict(re.findall(r"([a-zA-Z_]\w*)\s*=\s*([-\w\.]+)", rest))
                # coerce numerics
                for k,v in list(args.items()):
                    try:
                        args[k] = int(v) if re.match(r"^-?\d+$", v) else float(v)
                    except Exception:
                        pass
                acts.append({"type": typ, "args": args})
        t = {
            "goal": goal.group(1).strip() if goal else "",
            "actions": acts or [{"type":"noop","args":{}}],
            "expected_observation": exp.group(2).strip() if exp else "",
            "stop_signal": (stop and stop.group(2).strip().lower() in {"true","yes","1","stop"}) or False
        }
        thoughts.append(_normalize_thought(t))
    return thoughts

def parse_thoughts_from_text(raw: str) -> List[Dict[str, Any]]:
    s = _strip_fences(_norm_quotes(raw)).strip()
    s = _remove_trailing_commas(s)

    # prefer array extraction
    arr = _extract_first_json_array(s)

    # 1) strict JSON
    try:
        data = json.loads(arr)
        if not isinstance(data, list): data = [data]
        return [_normalize_thought(x) for x in data if isinstance(x, dict)]
    except Exception:
        pass

    # 2) Python-literal style
    try:
        data = ast.literal_eval(arr)
        if not isinstance(data, list): data = [data]
        out = []
        for x in data:
            if isinstance(x, dict):
                try: out.append(_normalize_thought(x))
                except Exception: continue
        if out: return out
    except Exception:
        pass

    # 3) last resort: parse bullets / key-value text
    fb = _fallback_from_bullets(s)
    return fb
