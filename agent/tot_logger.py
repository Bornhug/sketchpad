# trace_logger.py
from __future__ import annotations
import json, os, io, time
from pathlib import Path
from typing import List, Dict, Any, Optional

class TraceLogger:
    """
    Writes a chat-like JSON array where each element looks like:
      {
        "content": [{"type": "text", "text": "..."}],
        "role": "assistant" | "user"
      }
    and text blocks include headers like "# THOUGHT k:", "# ACTION k:", "OBSERVATION:", "ANSWER: ... TERMINATE"
    so it matches your reference JSON format.
    """
    def __init__(self,
                 out_path: Path,
                 autosave: bool = True,
                 indent: int = 4):
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._messages: List[Dict[str, Any]] = []
        self._step = 0
        self._indent = indent
        self._autosave = autosave

    # ---------- Public API ----------
    def next_step(self) -> int:
        s = self._step
        self._step += 1
        return s

    def record_user_request(self, text: str) -> None:
        self._append("user", text)

    def thought(self, k: int, text: str) -> None:
        self._append("assistant", f"# THOUGHT {k}:\n{text}")

    def action(self, k: int, code_or_text: str, language: str = "python") -> None:
        # If it's code, wrap in a fenced block just like your sample
        if "\n" in code_or_text or code_or_text.strip().startswith(("import ", "def ", "class ", "#")):
            payload = f"# ACTION {k}:\n```{language}\n{code_or_text}\n```"
        else:
            payload = f"# ACTION {k}:\n{code_or_text}"
        self._append("assistant", payload)

    def observation(self, text: str) -> None:
        self._append("user", f"OBSERVATION: {text}")

    def answer_and_terminate(self, answer_text: str) -> None:
        self._append("assistant", f"ANSWER: {answer_text} TERMINATE")

    def save(self) -> None:
        self._atomic_write(self._messages)

    # ---------- Helpers ----------
    def _append(self, role: str, text: str) -> None:
        msg = {
            "content": [{"type": "text", "text": text}],
            "role": role,
        }
        self._messages.append(msg)
        if self._autosave:
            self._atomic_write(self._messages)

    def _atomic_write(self, obj: Any) -> None:
        tmp = self.out_path.with_suffix(self.out_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=self._indent)
        os.replace(tmp, self.out_path)
