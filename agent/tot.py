# agent/tot.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple
import copy, heapq, time, os, json, shutil
from pathlib import Path
from PIL import Image
import io, base64

# >>> ADD
from tqdm import tqdm
# <<<



@dataclass
class ToTConfig:
    strategy: str = "bfs"          # "bfs" | "dfs" | "best_first"
    max_steps: int = 8
    beam_width: int = 4
    prune_threshold: float = 0.0
    eval_top_m: int = 1
    time_budget_s: float = 120.0
    # >>> ADD
    show_progress: bool = True                 # show a tqdm progress bar
    log_dir: Optional[str] = None              # where to dump traces; None = no logging
    log_images: bool = True                    # copy canvas snapshots if available
    # <<<

@dataclass
class Node:
    state: Any
    score: float
    step: int
    path: List[Any]
    done: bool = False
    # >>> ADD (optional id to link logs)
    node_id: Optional[int] = None
    # <<<

# >>> ADD: lightweight logger
class _ToTLogger:
    def __init__(self, log_dir: str, log_images: bool = True):
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.snap_dir = self.dir / "snapshots"
        if log_images:
            self.snap_dir.mkdir(exist_ok=True)
        self.log_images = log_images
        self.jl = open(self.dir / "nodes.jsonl", "w", encoding="utf-8")
        self._next_id = 0

    def close(self):
        try:
            self.jl.close()
        except Exception:
            pass

    def _snapshot(self, state: Any, step: int, node_id: int) -> Optional[str]:
        if not self.log_images:
            return None
        canvas = getattr(state, "canvas", None)
        if isinstance(canvas, str) and os.path.exists(canvas):
            dst = self.snap_dir / f"s{step:02d}_n{node_id}.png"
            try:
                shutil.copyfile(canvas, dst)
                return str(dst)
            except Exception:
                return None
        return None

    def log_node(self, *, parent_id: Optional[int], step: int, score: float,
                 thought: Optional[dict], state: Any, done: bool) -> int:
        nid = self._next_id
        self._next_id += 1
        snap = self._snapshot(state, step, nid)
        rec = {
            "node_id": nid,
            "parent_id": parent_id,
            "step": step,
            "score": score,
            "done": done,
            "thought": thought,            # None for root
            "canvas_snapshot": snap        # path or None
        }
        self.jl.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.jl.flush()
        return nid

    def write_best(self, best: Node):
        with open(self.dir / "best_path.json", "w", encoding="utf-8") as f:
            json.dump({
                "final_score": best.score,
                "steps": len(best.path),
                "thoughts": best.path,      # the reasoning trajectory
            }, f, indent=2, ensure_ascii=False)
# <<<

class ToTController:
    def __init__(
        self,
        propose_fn: Callable[[Any, int], List[Any]],
        apply_fn: Callable[[Any, Any], Any],
        eval_fn: Callable[[Any], float],
        is_terminal_fn: Callable[[Any], bool],
        cfg: ToTConfig,
    ):
        self.propose_fn = propose_fn
        self.apply_fn = apply_fn
        self.eval_fn = eval_fn
        self.is_terminal_fn = is_terminal_fn
        self.cfg = cfg
        # >>> ADD
        self.logger = _ToTLogger(cfg.log_dir, cfg.log_images) if cfg.log_dir else None
        # <<<

    def solve(self, init_state: Any) -> Node:
        t0 = time.time()
        frontier: List[Node] = []
        root = Node(state=init_state, score=self.eval_fn(init_state), step=0, path=[])
        # >>> ADD: log root
        if self.logger:
            root.node_id = self.logger.log_node(parent_id=None, step=0, score=root.score,
                                                thought=None, state=root.state, done=False)
        # <<<

        if self.cfg.strategy == "best_first":
            frontier = [(-root.score, 0, root)]
        else:
            frontier = [root]

        best = root
        visited = 0

        # >>> ADD: progress bar (track max depth reached)
        pbar = tqdm(total=self.cfg.max_steps, desc="ToT", leave=False, disable=not self.cfg.show_progress)
        last_depth_shown = 0
        # <<<

        try:
            while frontier and time.time() - t0 < self.cfg.time_budget_s:
                if self.cfg.strategy == "dfs":
                    cur = frontier.pop()
                elif self.cfg.strategy == "bfs":
                    cur = frontier.pop(0)
                else:
                    _, _, cur = heapq.heappop(frontier)

                visited += 1
                # update progress when we reach a deeper step
                if cur.step > last_depth_shown:
                    inc = min(cur.step, self.cfg.max_steps) - last_depth_shown
                    if inc > 0:
                        pbar.update(inc)
                        last_depth_shown += inc

                if self.is_terminal_fn(cur.state) or cur.step >= self.cfg.max_steps:
                    cur.done = True
                    if cur.score > best.score:
                        best = cur
                    # also log terminal state
                    if self.logger:
                        self.logger.log_node(parent_id=cur.node_id, step=cur.step, score=cur.score,
                                             thought=None, state=cur.state, done=True)
                    continue

                # inside ToTController.solve loop, replace:
                # thoughts = self.propose_fn(cur.state, self.cfg.beam_width)
                # with:
                try:
                    thoughts = self.propose_fn(cur.state, self.cfg.beam_width)
                    if not isinstance(thoughts, list): thoughts = []
                except Exception as e:
                    thoughts = []

                children: List[Node] = []
                for τ in thoughts:
                    s_next = self.apply_fn(cur.state, τ)

                    # later, before scoring each child:
                    try:
                        sc = self.eval_fn(s_next)
                        sc = float(sc) if sc is not None else 0.0
                    except Exception:
                        sc = 0.0

                    if sc >= self.cfg.prune_threshold:
                        child = Node(
                            state=s_next, score=sc, step=cur.step + 1,
                            path=cur.path + [τ], done=False
                        )
                        # >>> ADD: log child and assign id
                        if self.logger:
                            child.node_id = self.logger.log_node(parent_id=cur.node_id, step=child.step,
                                                                 score=child.score, thought=τ,
                                                                 state=child.state, done=False)
                        # <<<
                        children.append(child)
                        if sc > best.score:
                            best = child

                children.sort(key=lambda n: n.score, reverse=True)
                children = children[:self.cfg.eval_top_m]

                if self.cfg.strategy == "dfs":
                    frontier.extend(children)
                elif self.cfg.strategy == "bfs":
                    frontier.extend(children)
                else:
                    for c in children:
                        heapq.heappush(frontier, (-c.score, visited, c))
        finally:
            # >>> ADD: close progress bar + write best trace + close logger
            try:
                # ensure bar reaches max if we ended early; optional:
                # pbar.update(max(0, self.cfg.max_steps - last_depth_shown))
                pbar.close()
            except Exception:
                pass
            if self.logger:
                self.logger.write_best(best)
                self.logger.close()
            # <<<
        return best
