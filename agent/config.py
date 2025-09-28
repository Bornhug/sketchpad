import os

# set up the agent
MAX_REPLY = 10

# set up the LLM for the agent
os.environ['OPENAI_API_KEY'] = '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'
os.environ["AUTOGEN_USE_DOCKER"] = "False"
llm_config={"cache_seed": None, "temperature": 0.0, "config_list": [{"model": "gpt-4o-mini",  "api_key": os.environ.get("OPENAI_API_KEY")}]}


# use this after building your own server. You can also set up the server in other machines and paste them here.
SOM_ADDRESS = "http://localhost:8080/"
GROUNDING_DINO_ADDRESS = "http://localhost:8081/"
DEPTH_ANYTHING_ADDRESS = "http://localhost:8082/"
