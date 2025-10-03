from main import run_agent

# # run a example for graph max flow. save the execution trace, answer, and usage summary under outputs/graph_maxflow
# run_agent(
#     "/mnt/c/Users/apex/Code/Python/8539/sketchpad/tasks/graph_maxflow/5",
#     "/home/apex/sketchpad_runs/graph_max_flow",
#     task_type="math",
#     task_name="graph_maxflow",
# )

# run a example for geometry. save the execution trace, answer, and usage summary under outputs/geometry
run_agent(
    "/mnt/c/Users/apex/Code/Python/8539/sketchpad/tasks/geometry/1053",
    "/home/apex/sketchpad_runs/geometry",
    task_type="geo",
    task_name="geometry",
    search_mode = "tot",           
    tot_strategy = "bfs",             
    tot_branch = 3,
    tot_beam = 2,
    tot_steps = 3
)

