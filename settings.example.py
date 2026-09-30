# Copy this file to settings.py and set the IP of your Ollama server.
# settings.py is ignored by git, so your local IP stays out of the repo.
IP = "192.168.x.x"

# Models to test, one after another. Fill it with every model downloaded on
# the server by running: python3 query.py --update-models
# Comment out the ones you don't want to run.
MODELS = [
    "qwen3.5:9b",
]
