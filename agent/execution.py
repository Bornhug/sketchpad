import os, sys
import pickle
from autogen.coding import CodeBlock
from autogen.coding.jupyter import JupyterCodeExecutor, LocalJupyterServer

import ast, re


# add the tools directory to the path
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


# for each dialogue, we will have a new code executor
class CodeExecutor:
    def __init__(
        self,
        working_dir: str = "",
        use_vision_tools: bool = False,
        ):
        self.working_dir = working_dir

        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir, exist_ok=True)

        # set up the server
        self.server = LocalJupyterServer()
        #
        # set up the jupyter executor
        self.executor = JupyterCodeExecutor(self.server, output_dir=self.working_dir)

        # initialize the environment
        self.init_env(use_vision_tools)

    def result_processor(self, result):
        # Change an IPythonCodeResult object to a string, and the list of files
        # If the execution failed, the string is the error message.
        # If the execution is successful, the string is the output of the code execution.
        # In the string, all embeded PIL images are replaced by their file paths, using html img tag.
        # The list of files are the paths of the images.

        # process error message
        def parse_error_message(error):
            # Find the index where the list starts, indicated by `['`
            list_start_index = error.find("['")

            # The first part before the list is the initial error message
            initial_error = error[:list_start_index].strip()

            # The second part is the list of strings, which starts from `['` and goes to the end of the string
            traceback_list_str = error[list_start_index:]

            # Use ast.literal_eval to safely evaluate the string representation of the list
            # This method is safer than eval and can handle Python literals properly
            try:
                traceback_list = ast.literal_eval(traceback_list_str)
            except SyntaxError as e:
                print("Error parsing the list: ", e)
                traceback_list = []

            # Remove ANSI escape sequences
            ansi_escape = re.compile(r'\x1b\[.*?m')
            traceback_list = [ansi_escape.sub('', line) for line in traceback_list]

            return initial_error + "\n\n" + "\n".join(traceback_list)


        exit_code = result.exit_code

        file_paths = result.output_files
        output_str = result.output
        output_lines = output_str.split("\n")

        if len(file_paths) > 0:
            output_lines = output_lines[:-2*len(file_paths)]

        # if execution succeeded, replace PIL images with their file paths
        if exit_code == 0:
            new_str = ""
            image_idx = 0

            for line in output_lines:
                if line.startswith("<PIL."):
                    if image_idx < len(file_paths):
                        new_str += f"<img src='{file_paths[image_idx]}'>"
                        image_idx += 1
                else:
                    new_str += line
                new_str += "\n"

            # add the remaining images
            for file_idx, file in enumerate(file_paths):
                if file_idx >= image_idx:
                    new_str += f"<img src='{file}'>"
                    new_str += "\n"

            return exit_code, new_str, file_paths

        # if execution failed, parse the error message
        else:
            error_msg = parse_error_message(output_str)
            return exit_code, error_msg, file_paths

    def execute(self, code: str):
        self.executor._jupyter_kernel_client = self.executor._jupyter_client.get_kernel_client(self.executor._kernel_id)
        execution_result = self.executor.execute_code_blocks(
            code_blocks=[
                CodeBlock(language="python",
                        code=code),
            ]
        )
        ret = self.result_processor(execution_result)
        return ret

    def init_env(self, use_vision_tools):
        init_code = ("import sys\n"
                     "from PIL import Image\n"
                     "from IPython.display import display\n"
                     f"parent_dir = '{parent_dir}'\n"
                     "if parent_dir not in sys.path:\n"
                     "    sys.path.insert(0, parent_dir)\n"
        )
        if use_vision_tools:
            init_code += "from tools import *\n"

        init_resp = self.execute(init_code)
        print(init_resp[1])


    def cleanup(self):
        self.server.stop()

# import os, sys
# import pickle
# import platform
# import ast, re
#
# # ── AutoGen imports (Embedded IPython works on Windows)
# from autogen.coding.jupyter import EmbeddedIPythonCodeExecutor
# # If you want to keep CodeBlock-style execution, uncomment next line and use the alternate execute() shown below.
# from autogen.coding import CodeBlock
#
# # Add the tools directory to the path
# parent_dir = os.path.dirname(os.path.abspath(__file__))
# if parent_dir not in sys.path:
#     sys.path.insert(0, parent_dir)
#
# # Allow configuring a specific ipykernel (e.g., one you created named "sketchpad")
# KERNEL_NAME = os.getenv("IPYKERNEL_NAME", "python3")
#
#
# class CodeExecutor:
#     def __init__(self, working_dir: str = "", use_vision_tools: bool = False):
#         self.working_dir = working_dir or os.getcwd()
#         os.makedirs(self.working_dir, exist_ok=True)
#
#         # Windows-safe: embedded IPython (stateful, no kernel-gateway needed)
#         # Some AutoGen versions don't accept output_dir; pass only supported args.
#         self.executor = EmbeddedIPythonCodeExecutor(kernel_name=KERNEL_NAME, timeout=120)
#
#         # Ensure code runs inside the task folder
#         os.chdir(self.working_dir)
#
#         # initialize the environment inside the kernel
#         self.init_env(use_vision_tools)
#
#     def result_processor(self, result):
#         """Turn an IPython execution result into (exit_code, text, file_paths)."""
#
#         # Defensive access in case attributes differ across versions
#         exit_code = getattr(result, "exit_code", 0)
#         file_paths = getattr(result, "output_files", []) or []
#         output_str = getattr(result, "output", "") or ""
#
#         def parse_error_message(error: str) -> str:
#             # Split "message + list_repr" style traces; fall back cleanly
#             list_start_index = error.find("['")
#             if list_start_index == -1:
#                 return error.strip()
#
#             initial_error = error[:list_start_index].strip()
#             traceback_list_str = error[list_start_index:]
#
#             try:
#                 traceback_list = ast.literal_eval(traceback_list_str)
#             except Exception:
#                 traceback_list = [traceback_list_str]
#
#             # Remove ANSI escapes
#             ansi_escape = re.compile(r'\x1b\[.*?m')
#             traceback_list = [ansi_escape.sub('', line) for line in traceback_list]
#             return initial_error + "\n\n" + "\n".join(traceback_list)
#
#         # Success path: replace inline PIL prints with <img> tags
#         output_lines = output_str.split("\n")
#         if file_paths:
#             # Some executors echo file paths at the end; trim if needed (heuristic)
#             # Keep as-is if your executor doesn't append these.
#             pass
#
#         if exit_code == 0:
#             new_str, image_idx = "", 0
#             for line in output_lines:
#                 if line.startswith("<PIL."):
#                     if image_idx < len(file_paths):
#                         new_str += f"<img src='{file_paths[image_idx]}'>\n"
#                         image_idx += 1
#                 else:
#                     new_str += line + "\n"
#             # Append any remaining files
#             for file_idx, file in enumerate(file_paths):
#                 if file_idx >= image_idx:
#                     new_str += f"<img src='{file}'>\n"
#             return exit_code, new_str, file_paths
#         else:
#             return exit_code, parse_error_message(output_str), file_paths
#
#     def execute(self, code: str):
#         # # Embedded executor supports direct code execution
#         # execution_result = self.executor.execute(code)
#         # return self.result_processor(execution_result)
#
#         # --- Alternate (if you prefer CodeBlock batches) ---
#         execution_result = self.executor.execute_code_blocks(
#              code_blocks=[CodeBlock(language="python", code=code)]
#          )
#         return self.result_processor(execution_result)
#
#     def init_env(self, use_vision_tools: bool):
#         init_code = (
#             "import sys\n"
#             "from PIL import Image\n"
#             "from IPython.display import display\n"
#             f"parent_dir = {parent_dir!r}\n"
#             "if parent_dir not in sys.path:\n"
#             "    sys.path.insert(0, parent_dir)\n"
#         )
#         if use_vision_tools:
#             init_code += "from tools import *\n"
#
#         exit_code, out, files = self.execute(init_code)
#         # Print the init output (or error) to the host console for visibility
#         print(out)
#
#     def cleanup(self):
#         # No LocalJupyterServer to stop; shut down the embedded kernel if supported
#         shutdown = getattr(self.executor, "shutdown", None)
#         if callable(shutdown):
#             try:
#                 shutdown()
#             except Exception:
#                 pass



