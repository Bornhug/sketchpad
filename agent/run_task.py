from main import run_agent
import os, glob, argparse
from tqdm import tqdm

import os
from pathlib import Path
from tot_logger import TraceLogger



def run_task(task, output_dir, task_type="vision", task_name=None, **tot_kwargs):
    all_task_instances = glob.glob(
        f"../tasks/{task}/processed/*/" if task_type == "vision" else f"../tasks/{task}/*/"
    )
    output_dir = os.path.join(output_dir, task)

    for task_instance in tqdm(all_task_instances):
        print(f"Running task instance: {task_instance}")
        run_agent(task_instance, output_dir, task_type=task_type, task_name=task_name, **tot_kwargs)
        #break  # remove this 'break' to process all instances


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str,
                        choices=["vstar", "blink_viscorr", "blink_semcorr", "blink_depth", "blink_jigsaw",
                                 "blink_spatial", "mmvp", "geometry", "graph_connectivity", "graph_isomorphism",
                                 "graph_maxflow", "math_convexity", "math_parity", "winner_id"])
    parser.add_argument(
        "--instance",
        type=str,
        default=None,
        help="Optional path to a single task instance; used for naming outputs"
    )
    # >>> ToT flags
    parser.add_argument("--search_mode", choices=["single", "tot"], default="single")
    parser.add_argument("--tot_strategy", choices=["bfs", "dfs", "best_first"], default="bfs")
    parser.add_argument("--tot_branch", type=int, default=3)
    parser.add_argument("--tot_beam", type=int, default=2)
    parser.add_argument("--tot_steps", type=int, default=3)

    args = parser.parse_args()

    instance_name = None
    if args.instance:
        instance_name = Path(args.instance.rstrip('/\\')).name

    task_name = instance_name or args.task  # whatever you already parse
    out_dir = Path(os.environ.get("SKETCHPAD_OUT", "logs"))
    trace_path = out_dir / task_name / "trace.json"
    logger = TraceLogger(trace_path, autosave=True)

    # # (Optional) record the initial user request block if you want it in the file:
    # logger.record_user_request(f"USER REQUEST #: {your_user_request_string_here}")

    if args.task in ["vstar", "blink_viscorr", "blink_semcorr", "blink_depth", "blink_jigsaw", "blink_spatial", "mmvp"]:
        task_type, task_name = "vision", None
    elif args.task == "geometry":
        task_type, task_name = "geo", None
    else:
        task_type, task_name = "math", args.task

    out_root = os.environ.get("SKETCHPAD_OUT", "outputs")
    run_task(
        args.task, out_root, task_type=task_type, task_name=task_name,
        search_mode=args.search_mode, tot_strategy=args.tot_strategy,
        tot_branch=args.tot_branch, tot_beam=args.tot_beam, tot_steps=args.tot_steps
    )

# from main import run_agent
# import os, glob, argparse
# from tqdm import tqdm
#
# def run_task(task, output_dir, task_type="vision", task_name=None):
#     all_task_instances = glob.glob(f"../tasks/{task}/processed/*/" if task_type == "vision" else f"../tasks/{task}/*/")
#     output_dir = os.path.join(output_dir, task)
#
#     for task_instance in tqdm(all_task_instances):
#         print(f"Running task instance: {task_instance}")
#         run_agent(task_instance, output_dir, task_type=task_type, task_name=task_name)
#         break
#
# if __name__ == "__main__":
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--task", type=str, choices=["vstar", "blink_viscorr", "blink_semcorr", "blink_depth",
#                                                     "blink_jigsaw", "blink_spatial", "mmvp",
#                                                     "geometry",
#                                                     "graph_connectivity", "graph_isomorphism", "graph_maxflow",
#                                                     "math_convexity", "math_parity", "winner_id"], help="The task name")
#     args = parser.parse_args()
#
#     if args.task in ["vstar", "blink_viscorr", "blink_semcorr", "blink_depth","blink_jigsaw", "blink_spatial", "mmvp",]:
#         task_type = "vision"
#         task_name = None
#
#     elif args.task in ["geometry"]:
#         task_type = "geo"
#         task_name = None
#
#     else:
#         task_type = "math"
#         task_name = args.task
#
#     out_root = os.environ.get("SKETCHPAD_OUT", "outputs")
#     run_task(args.task, out_root, task_type=task_type, task_name=task_name)
