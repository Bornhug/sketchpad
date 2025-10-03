# agent/tot_parse.py
from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List


SMART_QUOTES = {
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u201f": '"',
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u2014": "-",
    "\u2013": "-",
}


def _norm_quotes(text: str) -> str:
    for bad, good in SMART_QUOTES.items():
        text = text.replace(bad, good)
    return text


def _strip_fences(text: str) -> str:
    # remove ```json ...``` or ``` ...```
    return re.sub(r"```(?:json|JSON)?\s*([\s\S]*?)```", r"\1", text, flags=re.MULTILINE)


def _remove_trailing_commas(text: str) -> str:
    # ,] or ,}  -> ] or }
    return re.sub(r",\s*([}\]])", r"\1", text)


def _extract_first_json_array(text: str) -> str:
    match = re.search(r"\[[\s\S]*\]", text)  # greedy to the last closing bracket
    return match.group(0) if match else text


def _normalize_action(action: Any) -> Dict[str, Any]:
    if isinstance(action, dict):
        if 'code' in action and not action.get('type'):
            return {'type': 'python', 'code': str(action['code'])}
        action_type = action.get('type') or action.get('name') or action.get('op')
        if action_type:
            if str(action_type).lower() in {'python', 'code'} and 'code' in action:
                return {'type': 'python', 'code': str(action['code'])}
            return {'type': str(action_type), 'args': dict(action.get('args', {}))}
        if 'code' in action:
            return {'type': 'python', 'code': str(action['code'])}
        raise ValueError('action missing type/name/op')
    # fallback for bare strings or other literals
    return {'type': str(action), 'args': {}}


def _normalize_thought_dict(thought: Dict[str, Any]) -> Dict[str, Any]:
    goal = thought.get('goal') or thought.get('intent') or thought.get('thought') or ''
    rationale = thought.get('rationale') or thought.get('reasoning') or goal
    actions = thought.get('actions') or thought.get('plan') or []
    if not isinstance(actions, list):
        actions = [actions]

    norm_actions: List[Dict[str, Any]] = []
    for action in actions:
        try:
            norm_actions.append(_normalize_action(action))
        except Exception:
            continue

    code = thought.get('code')
    if code is not None:
        norm_actions.append({'type': 'python', 'code': str(code)})

    exp = thought.get('expected_observation') or thought.get('expected') or ''
    stop = thought.get('stop_signal') or thought.get('stop') or False
    if isinstance(stop, (bool, int)):
        stop_flag = bool(stop)
    else:
        stop_flag = str(stop).strip().lower() in {'true', 'yes', '1'}

    return {
        'goal': str(goal),
        'rationale': str(rationale) if rationale is not None else '',
        'actions': norm_actions,
        'expected_observation': str(exp),
        'stop_signal': stop_flag,
    }


def _normalize_from_string(text: str) -> Dict[str, Any]:
    return {
        'goal': text,
        'rationale': text,
        'actions': [],
        'expected_observation': '',
        'stop_signal': False,
    }


def _fallback_from_bullets(raw: str) -> List[Dict[str, Any]]:
    thoughts: List[Dict[str, Any]] = []
    blocks = re.split(r"\n\s*\d+[\).\]]\s*", "\n" + raw.strip())
    for block in blocks:
        if not block.strip():
            continue
        goal_match = re.search(r"(?i)goal\s*:\s*(.*)", block)
        exp_match = re.search(r"(?i)(expected|expected_observation)\s*:\s*(.*)", block)
        stop_match = re.search(r"(?i)stop(_signal)?\s*:\s*(.*)", block)

        acts: List[Dict[str, Any]] = []
        for line in re.findall(r"(?:^|\n)\s*[-*]\s*(.*)", block):
            line = line.strip()
            if not line:
                continue
            name_match = re.match(r"([a-zA-Z_][\w]*)\s*(.*)", line)
            if not name_match:
                continue
            typ, rest = name_match.group(1), name_match.group(2)
            args = dict(re.findall(r"([a-zA-Z_]\w*)\s*=\s*([-\w\.]+)", rest))
            for key, val in list(args.items()):
                try:
                    args[key] = int(val) if re.match(r"^-?\d+$", val) else float(val)
                except Exception:
                    pass
            acts.append({'type': typ, 'args': args})

        t = {
            'goal': goal_match.group(1).strip() if goal_match else '',
            'actions': acts or [{'type': 'noop', 'args': {}}],
            'expected_observation': exp_match.group(2).strip() if exp_match else '',
            'stop_signal': (stop_match and stop_match.group(2).strip().lower() in {'true', 'yes', '1', 'stop'}) or False,
        }
        thoughts.append(_normalize_thought_dict(t))
    return thoughts


def parse_thoughts_from_text(raw: str) -> List[Dict[str, Any]]:
    s = _strip_fences(_norm_quotes(raw or '')).strip()
    s = _remove_trailing_commas(s)

    arr = _extract_first_json_array(s)

    # 1) strict JSON
    try:
        data = json.loads(arr)
        if not isinstance(data, list):
            data = [data]
        out: List[Dict[str, Any]] = []
        for item in data:
            if isinstance(item, dict):
                out.append(_normalize_thought_dict(item))
            elif isinstance(item, str):
                out.append(_normalize_from_string(item.strip()))
        if out:
            return out
    except Exception:
        pass

    # 2) Python-literal style
    try:
        data = ast.literal_eval(arr)
        if not isinstance(data, list):
            data = [data]
        out: List[Dict[str, Any]] = []
        for item in data:
            if isinstance(item, dict):
                try:
                    out.append(_normalize_thought_dict(item))
                except Exception:
                    continue
            elif isinstance(item, str):
                out.append(_normalize_from_string(item.strip()))
        if out:
            return out
    except Exception:
        pass

    # 3) last resort: parse bullets / key-value text
    return _fallback_from_bullets(s)

