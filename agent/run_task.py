# run_task.py
# was: from main import run_agent
from main import run_agent

import os
import glob
import argparse
from pathlib import Path
from tqdm import tqdm


def find_instances(task: str, task_type: str, instance: str ) -> list[str]:
    """Return a list of task instance directories to run."""
    if instance:
        return [instance.rstrip("/\\")]
    pattern = (
        f"../tasks/{task}/processed/*/" if task_type == "vision"
        else f"../tasks/{task}/*/"
    )
    # Normalize to remove trailing slashes
    return [p.rstrip("/\\") for p in glob.glob(pattern)]


def run_task(task: str, output_root: str, task_type: str = "vision", task_name: str  = None,
             instances: list[str] = None, **tot_kwargs):
    """Run one or more instances of a task."""
    output_dir = os.path.join(output_root, task)
    os.makedirs(output_dir, exist_ok=True)

    all_instances = instances or find_instances(task, task_type, None)
    if not all_instances:
        print(f"[run_task] No instances found for task={task} (type={task_type}).")
        return

    for task_instance in tqdm(all_instances):
        print(f"Running task instance: {task_instance}")
        run_agent(task_instance, output_dir, task_type=task_type, task_name=task_name, **tot_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Visual Sketchpad tasks.")
    parser.add_argument(
        "--task",
        type=str,
        choices=[
            "vstar", "blink_viscorr", "blink_semcorr", "blink_depth", "blink_jigsaw",
            "blink_spatial", "mmvp", "geometry",
            "graph_connectivity", "graph_isomorphism", "graph_maxflow",
            "math_convexity", "math_parity", "winner_id"
        ],
        required=True,
        help="The task name",
    )

    # Optional: run a single instance directory explicitly
    parser.add_argument(
        "--instance",
        type=str,
        default=None,
        help="Path to a single task instance dir (e.g., ../tasks/geometry/1029). "
             "If omitted, runs all instances for --task."
    )

    # ToT flags
    parser.add_argument("--search_mode", choices=["single", "tot"], default="single")
    parser.add_argument("--tot_strategy", choices=["bfs", "dfs", "best_first"], default="bfs")
    parser.add_argument("--tot_branch", type=int, default=3)
    parser.add_argument("--tot_beam", type=int, default=4)
    parser.add_argument("--tot_steps", type=int, default=8)

    args = parser.parse_args()

    # Map task to type/name
    if args.task in ["vstar", "blink_viscorr", "blink_semcorr", "blink_depth", "blink_jigsaw", "blink_spatial", "mmvp"]:
        task_type, task_name = "vision", None
    elif args.task == "geometry":
        task_type, task_name = "geo", None
    else:
        task_type, task_name = "math", args.task

    out_root = os.environ.get("SKETCHPAD_OUT", "outputs")

    # Build instance list (single vs. globbed)
    instances = find_instances(args.task, task_type, args.instance)

    run_task(
        args.task,
        out_root,
        task_type=task_type,
        task_name=task_name,
        instances=instances,
        search_mode=args.search_mode,
        tot_strategy=args.tot_strategy,
        tot_branch=args.tot_branch,
        tot_beam=args.tot_beam,
        tot_steps=args.tot_steps,
    )
