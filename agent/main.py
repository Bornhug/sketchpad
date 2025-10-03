import json
import os
import argparse
import shutil
import textwrap

from agent import SketchpadUserAgent
from multimodal_conversable_agent import MultimodalConversableAgent
from prompt import ReACTPrompt, MathPrompt, GeoPrompt, python_codes_for_images_reading, MULTIMODAL_ASSISTANT_MESSAGE
from parse import Parser
from execution import CodeExecutor
from utils import custom_encoder
from config import MAX_REPLY, llm_config

from dataclasses import dataclass
from typing import Any, List


# was: from agent.tot import ToTController, ToTConfig
from tot import ToTController, ToTConfig
# was: from agent.tot_prompts import propose_thoughts as tot_propose_thoughts
from tot_prompts import propose_thoughts as tot_propose_thoughts
from PIL import Image
from tot_logger import TraceLogger


def _exec_norm(executor, code: str, *, logger: TraceLogger  = None, step: int  = None, language: str = "python"):
    if logger is not None:
        k = step if step is not None else logger.next_step()
        logger.action(k, code, language=language)
    res = executor.execute(code)
    if isinstance(res, (tuple, list)):
        rc = res[0] if len(res) > 0 else 1
        out = res[1] if len(res) > 1 else ""
        err = res[2] if len(res) > 2 else ""
    elif isinstance(res, dict):
        rc = res.get("rc", 1); out = res.get("out", ""); err = res.get("err", "")
    else:
        rc, out, err = 0, str(res), ""

    if logger is not None:
        # Keep the observation compact to avoid huge files
        parts = []
        parts.append(f"rc={rc}")
        if out:
            parts.append(f"out={out[:500]}{'…' if len(out) > 500 else ''}")
        if err:
            parts.append(f"err={err[:500]}{'…' if len(err) > 500 else ''}")
        logger.observation("Execution finished: " + "; ".join(parts))

    return rc, out, err



# >>> ADDED for ToT
@dataclass
class VState:
    question: str
    text_trace: str          # running rationale
    canvas: Any              # PIL.Image | bytes | path (string)
    progress_score: float = 0.0
    has_final_answer: bool = False
    answer: str = ""


def checks_terminate_message(msg):
    if isinstance(msg, str):
        return "TERMINATE" in msg
    elif isinstance(msg, dict) and 'content' in msg and isinstance(msg['content'], str):
        return "TERMINATE" in msg['content']
    else:
        return False


def _as_query_text(x) -> str:
    """Make sure we pass a string question to agents/ToT."""
    if isinstance(x, str):
        return x
    try:
        # Prefer common fields if present
        for k in ("Problem", "question", "input", "query", "prompt", "instruction"):
            if isinstance(x, dict) and k in x and isinstance(x[k], str):
                return x[k]
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)


def _as_image_list(v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    if isinstance(v, str):
        return [v]
    return []


def run_agent(
    task_input, output_dir, task_type="vision", task_name=None,
    search_mode: str = "single",           # "single" (current) or "tot"
    tot_strategy: str = "bfs",             # "bfs" | "dfs" | "best_first"
    tot_branch: int = 3,
    tot_beam: int = 4,
    tot_steps: int = 3
):
    """Run the Visual Sketchpad agent on one task instance.

    Args:
        task_input (str): a path to the task input directory
        output_dir (str): a path to the directory where the output will be saved
        task_type (str): Task type. Should be vision, math, or geo. Defaults to "vision".
        task_name (str, optional): Only needed for math tasks. Defaults to None.
    """

    # task type should be one of "vision", "math", "geo"
    assert task_type in ["vision", "math", "geo"]

    # create a directory for the task
    task_input = task_input.rstrip('/')
    task_directory = os.path.join(output_dir, os.path.basename(task_input))

    # copy the task input to the output directory
    os.makedirs(output_dir, exist_ok=True)
    shutil.copytree(
        task_input,
        task_directory,
        dirs_exist_ok=True,
        copy_function=shutil.copy  # uses copyfile + copymode (lighter than copy2)
    )

    prompt_payload = None
    query_text = ""
    images: List[str] = []

    if task_type == "vision":
        # test if vision tools are loaded
        try:
            from tools import som_client, gd_client, da_client  # noqa: F401
        except ImportError as e:
            raise ImportError("Vision tools are not loaded. Please install vision_experts.") from e

        task_metadata = json.load(open(os.path.join(task_input, "request.json"), "r", encoding="utf-8"))
        query_field = (
            task_metadata.get("query")
            or task_metadata.get("input")
            or task_metadata.get("question")
            or ""
        )
        images = _as_image_list(task_metadata.get("images") or task_metadata.get("image"))
        query_text = query_field or _as_query_text(task_metadata)
        prompt_payload = query_text

        prompt_generator = ReACTPrompt()
        parser = Parser()
        executor = CodeExecutor(working_dir=task_directory, use_vision_tools=True)

        # read all images, save them in image_1, image_2, ... as PIL images
        image_reading_codes = python_codes_for_images_reading(images)
        rc, out, err = _exec_norm(executor, image_reading_codes)
        if rc != 0:
            raise RuntimeError(f"Error loading images:\nSTDERR: {err}\nSTDOUT: {out}")

    elif task_type == "math":
        payload = json.load(open(os.path.join(task_input, "example.json"), "r", encoding="utf-8"))
        prompt_payload = payload
        query_text = _as_query_text(payload)
        images = []
        prompt_generator = MathPrompt(task_name)
        parser = Parser()
        executor = CodeExecutor(working_dir=task_directory)

        # Build (or reuse) the base diagram as the starting canvas
        tot_canvas_path = os.path.join(task_directory, "tot_canvas.png")
        if not os.path.isfile(tot_canvas_path):
            init_code = textwrap.dedent(f"""
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                from pathlib import Path
                import importlib.util
                import PIL.Image as PILImage

                task_dir = Path(r'''{task_input}''')
                out_path = Path(r'''{tot_canvas_path}''')

                # Try Python diagram scripts first
                for cand in ['diagram.py','draw.py','input.py','base.py']:
                    p = task_dir / cand
                    if p.exists():
                        spec = importlib.util.spec_from_file_location('base_mod', str(p))
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                        fig, ax = plt.subplots(figsize=(6,6), dpi=200)
                        if hasattr(mod, 'draw'): mod.draw(ax)
                        elif hasattr(mod, 'render'): mod.render(ax)
                        ax.set_aspect('equal', adjustable='box')
                        fig.tight_layout()
                        fig.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
                        plt.close(fig)
                        break
                else:
                    # Try static base image
                    for cand in ['base.png','input.png','diagram.png','image.png']:
                        p = task_dir / cand
                        if p.exists():
                            PILImage.open(p).save(out_path)
                            break
                    else:
                        # Fallback: visible grid (last resort)
                        fig, ax = plt.subplots(figsize=(6,6), dpi=200)
                        ax.set_aspect('equal', adjustable='box')
                        ax.grid(True, linewidth=0.8, alpha=0.4)
                        ax.set_xlim(0,1); ax.set_ylim(0,1)
                        fig.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
                        plt.close(fig)
            """)
            rc, out, err = _exec_norm(executor, init_code)
            if rc != 0:
                print("[WARN] Failed to build base canvas (math). Proceeding without it.")
                print("stderr:", err[:500] if isinstance(err, str) else err)
                print("stdout:", out[:500] if isinstance(out, str) else out)

    elif task_type == "geo":
        payload = json.load(open(os.path.join(task_input, "ex.json"), "r", encoding="utf-8"))
        prompt_payload = payload
        query_text = payload.get("problem_text") or _as_query_text(payload)
        images = []
        prompt_generator = GeoPrompt()
        parser = Parser()
        executor = CodeExecutor(working_dir=task_directory)

        # Build (or reuse) the base diagram as the starting canvas
        tot_canvas_path = os.path.join(task_directory, "tot_canvas.png")
        if not os.path.isfile(tot_canvas_path):
            init_code = textwrap.dedent(f"""
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                from pathlib import Path
                import importlib.util
                import PIL.Image as PILImage

                task_dir = Path(r'''{task_input}''')
                out_path = Path(r'''{tot_canvas_path}''')

                # Try Python diagram scripts first
                for cand in ['diagram.py','draw.py','input.py','base.py']:
                    p = task_dir / cand
                    if p.exists():
                        spec = importlib.util.spec_from_file_location('base_mod', str(p))
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                        fig, ax = plt.subplots(figsize=(6,6), dpi=200)
                        if hasattr(mod, 'draw'): mod.draw(ax)
                        elif hasattr(mod, 'render'): mod.render(ax)
                        ax.set_aspect('equal', adjustable='box')
                        fig.tight_layout()
                        fig.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
                        plt.close(fig)
                        break
                else:
                    # Try static base image
                    for cand in ['base.png','input.png','diagram.png','image.png']:
                        p = task_dir / cand
                        if p.exists():
                            PILImage.open(p).save(out_path)
                            break
                    else:
                        # Fallback: visible grid (last resort)
                        fig, ax = plt.subplots(figsize=(6,6), dpi=200)
                        ax.set_aspect('equal', adjustable='box')
                        ax.grid(True, linewidth=0.8, alpha=0.4)
                        ax.set_xlim(0,1); ax.set_ylim(0,1)
                        fig.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
                        plt.close(fig)
            """)
            rc, out, err = _exec_norm(executor, init_code)
            if rc != 0:
                print("[WARN] Failed to build base canvas (geo). Proceeding without it.")
                print("stderr:", err[:500] if isinstance(err, str) else err)
                print("stdout:", out[:500] if isinstance(out, str) else out)



    # >>> ADDED for ToT — build initial state S0
    def _pick_base_canvas(task_directory: str) -> str:
        candidates = ["image.png", "base.png", "diagram.png", "input.png", "image.jpg", "diagram.jpg"]
        for name in candidates:
            p = os.path.join(task_directory, name)
            if os.path.isfile(p):
                return p
        return os.path.join(task_directory, "tot_canvas.png")

    if task_type in ("geo", "math"):
        init_canvas = _pick_base_canvas(task_directory)
    else:
        init_canvas = images[0] if (task_type == "vision" and len(images) > 0) else os.path.join(task_directory, "tot_canvas.png")


    # =================================

    if prompt_payload is None:
        prompt_payload = query_text

    try:
        initial_prompt_text = prompt_generator.initial_prompt(prompt_payload, len(images))
    except Exception:
        initial_prompt_text = query_text or ""

    S0 = VState(question=initial_prompt_text or query_text or "", text_trace="", canvas=init_canvas)
    # ===== Trace logging setup =====
    trace_path = os.path.join(task_directory, "trace.json")
    logger = TraceLogger(trace_path, autosave=True)

    # record the initial user message/question in the trace
    logger.record_user_request(query_text or "")
    if initial_prompt_text:
        logger.system(initial_prompt_text)

    # simple local step counter for pairing THOUGHT/ACTION/OBSERVATION
    _trace_step = {"k": 0}

    def _next_k() -> int:
        k = _trace_step["k"];
        _trace_step["k"] += 1;
        return k

    user = SketchpadUserAgent(
        name="multimodal_user_agent",
        human_input_mode='NEVER',
        max_consecutive_auto_reply=MAX_REPLY,
        is_termination_msg=checks_terminate_message,
        prompt_generator=prompt_generator,
        parser=parser,
        executor=executor
    )

    # running the planning experiment
    all_messages = {}

    planner = MultimodalConversableAgent(
        name="planner",
        human_input_mode='NEVER',
        max_consecutive_auto_reply=MAX_REPLY,
        is_termination_msg=lambda x: False,
        system_message=MULTIMODAL_ASSISTANT_MESSAGE,
        llm_config=llm_config
    )

    if initial_prompt_text:
        try:
            formatted = planner._message_to_dict(initial_prompt_text)
            planner._oai_messages.setdefault(user, [])
            planner._oai_messages[user].append({
                "role": "user",
                "content": formatted["content"],
            })
        except Exception:
            planner._oai_messages.setdefault(user, [])
            planner._oai_messages[user].append({
                "role": "user",
                "content": initial_prompt_text,
            })

    # >>> ADDED for ToT — LLM completion adapter used by tot_prompts
    def llm_complete(prompt: str) -> str:
        """Share planner context with ToT completions so prompts stay consistent."""
        from copy import deepcopy

        messages = []
        system_msgs = getattr(planner, "_oai_system_message", [])
        if system_msgs:
            messages.extend(deepcopy(system_msgs))

        history = getattr(planner, "_oai_messages", {})
        if history:
            messages.extend(deepcopy(history.get(user, [])))

        prompt_dict = planner._message_to_dict(prompt)
        messages.append({
            "role": "user",
            "content": deepcopy(prompt_dict.get("content", []))
        })

        print("[ToT] llm_complete planner.client is None?", planner.client is None, flush=True)
        print("[ToT] llm_complete message count:", len(messages), flush=True)

        if planner.client is None:
            print("[ToT] planner has no client; returning empty string", flush=True)
            return ""

        try:
            ok, reply = planner.generate_oai_reply(messages=messages, sender=user)
            print("[ToT] planner.generate_oai_reply ->", ok, type(reply), flush=True)
            if ok:
                return reply if isinstance(reply, str) else reply.get("content", "")
        except Exception as e:
            print("[ToT] planner.generate_oai_reply failed:", repr(e), flush=True)

        tmp = MultimodalConversableAgent(
            name="tot_eval",
            human_input_mode='NEVER',
            max_consecutive_auto_reply=1,
            is_termination_msg=lambda x: False,
            system_message="You are a concise scoring assistant.",
            llm_config=llm_config
        )

        try:
            ok, reply = tmp.generate_oai_reply(messages=messages[-1:])
            print("[ToT] fallback generate_oai_reply ->", ok, type(reply), flush=True)
            if ok:
                return reply if isinstance(reply, str) else reply.get("content", "")
        except Exception as e:
            print("[ToT] fallback generate_oai_reply failed:", repr(e), flush=True)

        return ""

    # >>> ADDED for ToT — glue functions
    def propose_thoughts(state: VState, K: int) -> List[dict]:
        return tot_propose_thoughts(state, K, llm_complete)

    def apply_thought(state: VState, thought: dict) -> VState:
        """
        Apply one ToT node (a 'thought') to the current visual state by
        (1) logging THOUGHT,
        (2) optionally running an ACTION program,
        (3) logging OBSERVATION, and
        (4) returning the updated state.

        This version writes a canvas image to '<task_directory>/tot_canvas.png'
        so downstream code has a concrete artifact to read.
        """
        import os, json, textwrap
        from collections.abc import Mapping

        # ---- copy incoming state so we stay pure-functional
        new_state = VState(**{**state.__dict__})

        # ---- log THOUGHT (goal/why) ---------------------------------------------
        k = _next_k()
        goal = thought.get("goal", "")
        rationale_raw = str(thought.get("rationale", ""))
        stop_signal = bool(thought.get("stop_signal"))

        final_answer_line = None
        if "FINAL_ANSWER:" in rationale_raw:
            after = rationale_raw.split("FINAL_ANSWER:", 1)[1].strip()
            if after:
                final_answer_line = after.splitlines()[0].strip()

        rationale_clean = rationale_raw
        if not stop_signal and final_answer_line:
            rationale_clean = "\n".join(
                ln for ln in rationale_raw.splitlines()
                if "FINAL_ANSWER:" not in ln
            ).strip()

        thought_lines: list[str] = []
        if goal:
            thought_lines.append(f"[goal] {goal}")
        if rationale_clean:
            thought_lines.append(f"[why] {rationale_clean}")
        if stop_signal:
            thought_lines.append("[stop] true")
            if final_answer_line:
                thought_lines.append(f"FINAL_ANSWER: {final_answer_line}")

        if thought_lines:
            logger.thought(k, "\n".join(thought_lines))
            block = "\n".join(thought_lines)
            new_state.text_trace = (new_state.text_trace + "\n" + block).strip()
        else:
            logger.thought(k, "(no explicit goal/why provided)")

        # ---- normalize actions list ---------------------------------------------
        actions_in = thought.get("actions", []) or []
        actions: list[dict] = []
        for a in actions_in:
            if isinstance(a, Mapping):
                # keep dict actions as-is
                actions.append(dict(a))
            else:
                # allow bare strings as comments
                actions.append({"op": "comment", "text": str(a)})

        if stop_signal and final_answer_line:
            new_state.has_final_answer = True
            new_state.answer = final_answer_line

        # ---- no actions? still log ------------------------------------------------
        if not actions:
            logger.action(k, "No action needed.")
            logger.observation("No tool/code execution for this thought.")
            return new_state

        # Build self-contained program for CodeExecutor
        py = textwrap.dedent(f"""
            import os, shutil, json, math
            import numpy as np
            from PIL import Image
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle, Arc, Polygon

            ROOT = {task_directory!r}
            OUT = os.path.join(ROOT, 'tot_canvas.png')
            ACTIONS = {json.dumps(actions, ensure_ascii=False)}

            # 1) ensure base image copied into OUT on first draw
            cands = ['image.png','base.png','diagram.png','input.png','image.jpg','diagram.jpg']
            base = next((os.path.join(ROOT, n) for n in cands if os.path.exists(os.path.join(ROOT, n))), None)
            if (not os.path.exists(OUT)) and base:
                shutil.copyfile(base, OUT)

            # 2) load current canvas
            bg = Image.open(OUT).convert('RGB') if os.path.exists(OUT) else Image.new('RGB', (1024,1024), 'white')
            W, H = bg.size; dpi = 200
            fig = plt.figure(figsize=(W/dpi, H/dpi), dpi=dpi)
            ax = plt.axes([0,0,1,1])
            ax.imshow(bg, extent=[0,1,0,1], origin='lower')
            ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off'); ax.set_aspect('equal', adjustable='box')

            def _col(c): return c if isinstance(c,str) and c else 'black'
            def _lw(v):
                try: return float(v)
                except: return 2.0
            def _alpha(v):
                try: return float(v)
                except: return 1.0
            def _ls(s): return s if s in ('-','--',':','-.','') else '-'
            def _pt(p): return (float(p[0]), float(p[1]))

            def _clip01(p1,p2):
                x1,y1 = p1; x2,y2 = p2; dx,dy = x2-x1, y2-y1
                if dx==0 and dy==0: return None
                t0,t1 = 0.0,1.0
                for p,q in ((-dx,x1-0.0),(dx,1.0-x1),(-dy,y1-0.0),(dy,1.0-y1)):
                    if p==0:
                        if q<0: return None
                    else:
                        t = q/p
                        if p<0:
                            if t>t1: return None
                            if t>t0: t0=t
                        else:
                            if t<t0: return None
                            if t<t1: t1=t
                xa,ya = x1+t0*dx, y1+t0*dy; xb,yb = x1+t1*dx, y1+t1*dy
                return (xa,ya),(xb,yb)

            def line(p1,p2,c='black',lw=2,ls='-',a=1.0):
                p1,p2 = _pt(p1), _pt(p2); seg = _clip01(p1,p2)
                if seg is None: return
                (xa,ya),(xb,yb) = seg
                ax.plot([xa,xb],[ya,yb], color=_col(c), linewidth=_lw(lw), linestyle=_ls(ls), alpha=_alpha(a))
            def segment(p1,p2,c='black',lw=2,ls='-',a=1.0):
                x1,y1 = _pt(p1); x2,y2 = _pt(p2)
                ax.plot([x1,x2],[y1,y2], color=_col(c), linewidth=_lw(lw), linestyle=_ls(ls), alpha=_alpha(a))
            def ray(p1,p2,c='black',lw=2,ls='-',a=1.0):
                p1,p2 = _pt(p1), _pt(p2); dx,dy = p2[0]-p1[0], p2[1]-p1[1]
                if dx==0 and dy==0: return
                seg = _clip01(p1,(p1[0]+10*dx,p1[1]+10*dy))
                if seg is None: return
                (xa,ya),(xb,yb) = seg
                ax.plot([xa,xb],[ya,yb], color=_col(c), linewidth=_lw(lw), linestyle=_ls(ls), alpha=_alpha(a))
            def polyline(pts,c='black',lw=2,ls='-',a=1.0):
                P=[_pt(p) for p in pts]; ax.plot([p[0] for p in P],[p[1] for p in P], color=_col(c), linewidth=_lw(lw), linestyle=_ls(ls), alpha=_alpha(a))
            def polygon(pts,ec='black',fc=None,lw=2,a=1.0):
                P=[_pt(p) for p in pts]; poly=Polygon(P,closed=True, fill=fc is not None, edgecolor=_col(ec), facecolor=(_col(fc) if fc else None), linewidth=_lw(lw), alpha=_alpha(a)); ax.add_patch(poly)
            def point(p,r=0.006,c='black',ec='black',lw=1.5,a=1.0,label=None,dx=0.012,dy=0.012,fs=10):
                x,y = _pt(p); circ=Circle((x,y),r,facecolor=_col(c),edgecolor=_col(ec),linewidth=_lw(lw),alpha=_alpha(a),zorder=10); ax.add_patch(circ)
                if label: ax.text(x+dx,y+dy,str(label),fontsize=fs,color=_col(ec),alpha=_alpha(a),zorder=11)
            def text(p,t,fs=12,c='black',a=1.0,ha='left',va='bottom',rot=0):
                x,y=_pt(p); ax.text(x,y,str(t),fontsize=fs,color=_col(c),alpha=_alpha(a),ha=ha,va=va,rotation=float(rot),zorder=12)
            def circle(center,r,ec='black',lw=2,ls='-',a=1.0,fill=False,fc=None):
                cx,cy=_pt(center); c=Circle((cx,cy),float(r),fill=fill and fc is not None,edgecolor=_col(ec),facecolor=(_col(fc) if fc else None),linewidth=_lw(lw),alpha=_alpha(a),linestyle=_ls(ls)); ax.add_patch(c)
            def _circum(a,b,c):
                (x1,y1)=_pt(a); (x2,y2)=_pt(b); (x3,y3)=_pt(c); d=2*(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2))
                if abs(d)<1e-9: return None
                ux=((x1**2+y1**2)*(y2-y3)+(x2**2+y2**2)*(y3-y1)+(x3**2+y3**2)*(y1-y2))/d
                uy=((x1**2+y1**2)*(x3-x2)+(x2**2+y2**2)*(x1-x3)+(x3**2+y3**2)*(x2-x1))/d
                r=math.hypot(ux-x1,uy-y1); return ux,uy,r
            def circle3(a,b,c,ec='black',lw=2,a1=1.0,ls='-'):
                cc=_circum(a,b,c)
                if cc is None: return
                ux,uy,r=cc; circle((ux,uy),r,ec=ec,lw=lw,ls=ls,a=a1)
            def arc(center,r,start,end,ec='black',lw=2,a=1.0):
                cx,cy=_pt(center); ar=Arc((cx,cy),2*float(r),2*float(r),angle=0.0,theta1=float(start),theta2=float(end),edgecolor=_col(ec),linewidth=_lw(lw),alpha=_alpha(a)); ax.add_patch(ar)
            def angle_mark(a,o,b,r=0.06,ec='black',lw=2,a1=1.0,label=None,fs=10):
                import numpy as _np; (ax_,ay_)=_pt(a); (ox,oy)=_pt(o); (bx_,by_)=_pt(b); va=_np.array([ax_-ox,ay_-oy]); vb=_np.array([bx_-ox,by_-oy])
                if _np.linalg.norm(va)<1e-9 or _np.linalg.norm(vb)<1e-9: return
                ta=math.degrees(math.atan2(va[1],va[0])); tb=math.degrees(math.atan2(vb[1],vb[0])); start, end = ta%360.0, tb%360.0
                if end<=start: end+=360.0
                arc((o[0],o[1]), r, start, end, ec=ec, lw=lw, a=a1)
                if label:
                    mid=math.radians((start+end)/2.0); lx,ly=ox+(r+0.03)*math.cos(mid), oy+(r+0.03)*math.sin(mid)
                    text((lx,ly), label, fs=fs, c=ec, a=a1, ha='center', va='center')
            def perp_through(p,a1,a2,c='black',lw=2,ls='--',a=1.0):
                (x1,y1)=_pt(a1); (x2,y2)=_pt(a2); (px,py)=_pt(p); dx,dy=x2-x1,y2-y1; q=(px-dy,py+dx); line((px,py),q,c,lw,ls,a)
            def parallel_through(p,a1,a2,c='black',lw=2,ls='--',a=1.0):
                (x1,y1)=_pt(a1); (x2,y2)=_pt(a2); (px,py)=_pt(p); dx,dy=x2-x1,y2-y1; q=(px+dx,py+dy); line((px,py),q,c,lw,ls,a)

            for act in ACTIONS:
                t = (act.get('type') or '').lower()
                A = act.get('args') or {{}}
                if t == 'line':           line(A.get('p1',[0,0]), A.get('p2',[1,1]), A.get('color','black'), A.get('lw',2), A.get('ls','-'), A.get('alpha',1.0))
                elif t in ('segment','line_segment'):
                    segment(A.get('p1',[0,0]), A.get('p2',[1,1]), A.get('color','black'), A.get('lw',2), A.get('ls','-'), A.get('alpha',1.0))
                elif t == 'ray':          ray(A.get('p1',[0,0]), A.get('p2',[1,1]), A.get('color','black'), A.get('lw',2), A.get('ls','-'), A.get('alpha',1.0))
                elif t in ('polyline','path'):
                    polyline(A.get('pts', []), A.get('color','black'), A.get('lw',2), A.get('ls','-'), A.get('alpha',1.0))
                elif t in ('polygon','poly'):
                    polygon(A.get('pts', []), A.get('edgecolor','black'), A.get('facecolor',None), A.get('lw',2), A.get('alpha',1.0))
                elif t in ('point','dot'):
                    point(A.get('p',[0.5,0.5]), A.get('r',0.006), A.get('color','black'), A.get('edgecolor','black'), A.get('lw',1.5), A.get('alpha',1.0),
                          A.get('label',None), A.get('label_dx',0.012), A.get('label_dy',0.012), A.get('fontsize',10))
                elif t == 'text':         text(A.get('p',[0.5,0.5]), A.get('text',''), A.get('fontsize',12), A.get('color','black'), A.get('alpha',1.0), A.get('ha','left'), A.get('va','bottom'), A.get('rotate',0))
                elif t == 'circle':       circle(A.get('center',[0.5,0.5]), A.get('r',0.2), A.get('edgecolor','black'), A.get('lw',2), A.get('ls','-'), A.get('alpha',1.0), A.get('fill',False), A.get('facecolor',None))
                elif t in ('circle3','circumcircle','circle_three_points'):
                    circle3(A.get('a',[0.2,0.5]), A.get('b',[0.7,0.5]), A.get('c',[0.5,0.8]), A.get('edgecolor','black'), A.get('lw',2), A.get('alpha',1.0), A.get('ls','-'))
                elif t in ('arc','angle_arc'):
                    arc(A.get('center',[0.5,0.5]), A.get('r',0.2), A.get('start',0), A.get('end',90), A.get('edgecolor','black'), A.get('lw',2), A.get('alpha',1.0))
                elif t in ('angle','angle_mark'):
                    angle_mark(A.get('a',[0.6,0.5]), A.get('o',[0.5,0.5]), A.get('b',[0.5,0.7]), A.get('r',0.06), A.get('edgecolor','black'), A.get('lw',2), A.get('alpha',1.0), A.get('label',None), A.get('fontsize',10))
                elif t in ('perpendicular','perp_through'):
                    perp_through(A.get('p',[0.5,0.5]), A.get('a1',[0.3,0.3]), A.get('a2',[0.7,0.3]), A.get('color','black'), A.get('lw',2), A.get('ls','--'), A.get('alpha',1.0))
                elif t in ('parallel','parallel_through'):
                    parallel_through(A.get('p',[0.5,0.5]), A.get('a1',[0.3,0.3]), A.get('a2',[0.7,0.3]), A.get('color','black'), A.get('lw',2), A.get('ls','--'), A.get('alpha',1.0))

            fig.savefig(OUT, dpi=dpi)
            plt.close(fig)
        """)

        rc, out, err = _exec_norm(executor, py)
        if rc != 0:
            raise RuntimeError(f"Drawing failed: {err or out}")

        # summarize execution for the trace
        obs_items: list[str] = ["rc=" + str(rc)]
        if out:
            clipped = out if len(out) <= 200 else out[:200] + "…"
            obs_items.append("stdout=" + clipped)
        if err:
            clipped_err = err if len(err) <= 200 else err[:200] + "…"
            obs_items.append("stderr=" + clipped_err)
        if actions:
            obs_items.append(f"actions={len(actions)}")
        logger.observation("; ".join(obs_items))

        new_state.canvas = os.path.join(task_directory, "tot_canvas.png")
        return new_state

    def eval_state(state: VState) -> float:
        # Simple LM-based scalar scoring (0..1). Replace with task heuristics when you have them.
        score_prompt = (
            "Score the progress toward solving the task (0..1). "
            "Only output a float.\n\n"
            f"Question:\n{state.question}\n\nRationale so far:\n{state.text_trace}\n"
        )
        try:
            s = llm_complete(score_prompt).strip()
            return float(s.split()[0])
        except Exception:
            return 0.0

    def is_terminal(state: VState) -> bool:
        # Stop if you’ve written a final answer token into the rationale, or hit a high score.
        if "FINAL_ANSWER:" in state.text_trace:
            return True
        return state.progress_score >= 0.999

    try:
        if search_mode == "tot":
            cfg = ToTConfig(
                strategy=tot_strategy,
                max_steps=tot_steps,
                beam_width=tot_branch,
                eval_top_m=tot_beam,
                prune_threshold=0.0,
                time_budget_s=180.0,
                show_progress=True,
                log_dir=os.path.join(task_directory, "tot_trace"),
                log_images=True,
            )
            controller = ToTController(
                propose_fn=propose_thoughts,
                apply_fn=apply_thought,
                eval_fn=eval_state,
                is_terminal_fn=is_terminal,
                cfg=cfg
            )
            best = controller.solve(S0)

            # ---- TRACE: final ANSWER and TERMINATE ---------------------------------------
            try:
                final_answer = None

                # Prefer to extract from the text_trace if you append 'FINAL_ANSWER: ...'
                tt = getattr(best.state, "text_trace", "") or ""
                if "FINAL_ANSWER:" in tt:
                    final_answer = tt.split("FINAL_ANSWER:", 1)[1].strip().splitlines()[0]

                # Fallbacks: explicit field or any last-resort text you store on state
                if not final_answer:
                    # Some pipelines store it here:
                    final_answer = getattr(best.state, "answer", "") or None
                if not final_answer:
                    # Last fallback: try a heuristic from the end of text_trace
                    lines = [ln.strip() for ln in tt.splitlines() if ln.strip()]
                    if lines:
                        for candidate in reversed(lines):
                            if candidate not in {"[goal]", "[why]"}:
                                final_answer = candidate
                                break

                if final_answer:
                    logger.answer_and_terminate(final_answer)
                else:
                    logger.answer_and_terminate("(no explicit FINAL_ANSWER found)")
            except Exception as _e:
                # We still want a terminating record even if parsing fails.
                logger.answer_and_terminate("(failed to extract final answer)")

            # Ensure the JSON file is flushed to disk
            logger.save()

            # synthesize a minimal message log for parity with your outputs
            all_messages = {
                "tot": {
                    "strategy": cfg.strategy,
                    "branch": cfg.beam_width,
                    "beam": cfg.eval_top_m,
                    "max_steps": cfg.max_steps
                },
                "final_rationale": best.state.text_trace,
                "final_canvas": os.path.relpath(best.state.canvas, task_directory) if isinstance(best.state.canvas, str) else "in_memory",
                "score": best.score
            }
        else:
            # original single-trajectory behavior
            user.initiate_chat(
                planner,
                n_image=len(images),
                task_id="testing_case",
                message=prompt_payload,
                log_prompt_only=False,
            )
            all_messages = planner.chat_messages[user]

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)
        all_messages = {'error': getattr(e, 'message', str(e))}

    # save the results
    with open(os.path.join(task_directory, "output.json"), "w", encoding="utf-8") as f:
        json.dump(all_messages, f, indent=4, default=custom_encoder, ensure_ascii=False)

    # usage summary (best-effort)
    usage_summary = {}
    try:
        usage_summary = {
            'total': getattr(planner.client, 'total_usage_summary', None),
            'actual': getattr(planner.client, 'actual_usage_summary', None)
        }
    except Exception:
        pass
    with open(os.path.join(task_directory, "usage_summary.json"), "w", encoding="utf-8") as f:
        json.dump(usage_summary, f, indent=4, ensure_ascii=False)

    # turn off server / cleanup
    try:
        user.executor.cleanup()
    except Exception:
        pass

    user.reset()
    planner.reset()
