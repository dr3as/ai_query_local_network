# Copy this file to settings.py and set the IP of your Ollama server.
# settings.py is ignored by git, so your local IP stays out of the repo.
IP = "192.168.x.x"

# Models to test, one after another. Fill it with every model downloaded on
# the server by running: python3 query.py --update-models
# Comment out the ones you don't want to run.
MODELS = [
    "qwen3.5:9b",
]

# Options sent to the model with every question. Remove a line to use the
# model's own default. More options you can add here:
#   "top_p": 0.9           Only pick from the most likely words that together make up
#                          90% of the probability. Lower = more focused answers.
#   "top_k": 40            Only pick from the 40 most likely next words.
#   "repeat_penalty": 1.1  Above 1.0 makes the model less likely to repeat itself.
#   "seed": 42             Same seed + same question = same answer. Good for
#                          fair comparisons between runs.
#   "stop": ["\n\n"]       Stop generating when one of these strings appears.
#   "num_gpu": 99          How many model layers to put on the GPU (0 = CPU only).
#   "num_thread": 8        CPU threads to use for the parts that run on the CPU.
# Full list: https://docs.ollama.com/modelfile (see "Valid Parameters and Values")
OPTIONS = {
    # Max tokens to generate. -1 = no limit, so thinking models aren't cut off
    # mid-thought. Set e.g. 512 to cap long answers.
    "num_predict": -1,
    # Context window: how many tokens of question + thinking + answer the model
    # can hold. Higher uses more VRAM. Raise it (e.g. 16384 or 32768) when piping
    # in big files; the model forgets the start of the text if it doesn't fit.
    "num_ctx": 8192,
    # Randomness, from 0 to about 2. Low (0-0.3) = focused and repeatable, good for
    # facts, math and code. High (0.8+) = more varied and creative.
    "temperature": 0.3,
}

# How long the model stays in VRAM after answering. 0 = unload right away, so
# several models can be tested one after another without filling up the GPU.
# Use e.g. "5m" or "1h" to keep it loaded, so repeated questions skip the load time.
KEEP_ALIVE = 0

# Seconds to wait for a model to answer before giving up.
TIMEOUT = 600

# query_history.py: max length of the conversation memory, in words. The memory is
# sent with every question, so longer = remembers more details but uses more of
# num_ctx and makes each question a bit slower.
HISTORY_SUMMARY_WORDS = 300

# query_history.py: how many of the latest exchanges are sent word for word. Older
# ones are folded into the memory. Higher = follow-up questions work better, but
# uses more of num_ctx. 0 = only the memory, nothing word for word.
HISTORY_RECENT_TURNS = 2
