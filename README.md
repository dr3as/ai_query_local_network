# ai_query_local_network
Script that query my local ollama install on my network, with features

## What it does

`query.py` sends a prompt to one or more models on an Ollama server on your local network, one model at a time. For each model it prints:

- the model's thinking (chain of thought), if it produces one
- the final answer
- performance metrics: tokens/sec, generated tokens, model load time, prompt evaluation time and total wall time

Each model is unloaded from VRAM right after it answers (`keep_alive=0`), so the models can be tested one after another without filling up the GPU.

## Requirements

- Python 3
- The `ollama` Python package: `pip install ollama`
- An Ollama server reachable on your network (default port `11434`)

## Setup

The server IP is kept in `settings.py`, which is ignored by git so your local IP isn't committed. Create it from the example:

```bash
cp settings.example.py settings.py
```

Then edit `settings.py` and set `IP` to the address of your Ollama server:

```python
IP = "192.168.1.50"
```

### Models

The models to test are listed in `MODELS` in `settings.py`. To fill the list with every model already downloaded on the server, run:

```bash
python3 query.py --update-models
```

This asks the server for its downloaded models, replaces the `MODELS` list in `settings.py` with them and exits without running any tests. Anything else in `settings.py` is kept. Comment out the models you don't want to run. Running `--update-models` again overwrites the list, including those comments.

## Usage

Set the `prompt` in `query.py`, then run:

```bash
python3 query.py
```

The model options (`num_ctx`, `temperature`, `num_predict`) are set in the `client.chat(...)` call in `query.py`.
