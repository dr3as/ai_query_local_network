# ai_query_local_network
Script that query my local ollama install on my network, with features

## What it does

`query.py` sends a question to one or more models on an Ollama server on your local network, one model at a time. For each model it prints, as it is being written:

- the model's thinking (chain of thought), if it produces one (hide it with `--nothinking`)
- the final answer
- with `--stats`: performance metrics (tokens/sec, generated tokens, model load time, prompt evaluation time and total wall time), plus a summary table comparing the models when more than one is run

By default each model is unloaded from VRAM right after it answers, so the models can be tested one after another without filling up the GPU (see `KEEP_ALIVE` below).

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

### Settings

| Setting | What it does |
|---|---|
| `IP` | Address of the Ollama server. |
| `MODELS` | The models to test (see below). |
| `OPTIONS` | Options sent to the model with every question, e.g. `temperature` and `num_ctx`. |
| `KEEP_ALIVE` | How long a model stays in VRAM after answering. `0` unloads it right away; `"5m"` or `"1h"` keeps it loaded, so repeated questions skip the load time. |
| `TIMEOUT` | Seconds to wait for a model to answer before giving up. |

`settings.example.py` explains each option and lists more you can add, such as `top_p`, `repeat_penalty` and `seed`. The main ones:

- `temperature`: randomness. Low (0–0.3) gives focused, repeatable answers for facts, math and code; high (0.8+) gives more varied, creative answers.
- `num_ctx`: how many tokens of question, thinking and answer the model can hold. Raise it when piping in big files, or the model loses the start of the text. Higher uses more VRAM.
- `num_predict`: max tokens to generate. `-1` means no limit, so thinking models aren't cut off mid-thought.
- `seed`: the same seed and question give the same answer, which makes comparisons fairer.

If `OPTIONS`, `KEEP_ALIVE` or `TIMEOUT` is missing from `settings.py`, the script uses the same defaults as the example.

### Models

The models to test are listed in `MODELS` in `settings.py`. To fill the list with every model already downloaded on the server, run:

```bash
python3 query.py --update-models
```

This asks the server for its downloaded models, replaces the `MODELS` list in `settings.py` with them and exits without running any tests. Anything else in `settings.py` is kept. Comment out the models you don't want to run. Running `--update-models` again overwrites the list, including those comments.

## Usage

Everything after the switches is the question:

```bash
python3 query.py What is 6+6
```

Put the question in quotes if it contains characters the shell treats specially, such as `?`, `*`, `'` or `!`:

```bash
python3 query.py --model=2 "What's 6+6?"
```

This asks every model in `MODELS`. To ask just some of them, list the models with their numbers and pick with `--model`:

```bash
python3 query.py --list-models
python3 query.py --model=2 What is 6+6          # model 2
python3 query.py --model=1,3,5 What is 6+6      # models 1, 3 and 5
python3 query.py --model=1-3 What is 6+6        # models 1 to 3
python3 query.py --model=qwen3.5:9b What is 6+6 # by name (doesn't have to be in MODELS)
```

The numbers follow the order of `MODELS` in `settings.py`, so they change if you edit the list or run `--update-models`.

### Switches

| Switch | What it does |
|---|---|
| `--model=N` | Run only these models: a number, numbers separated by commas, a range, or a model name. |
| `--list-models` | List the models in `settings.py` with their numbers. |
| `--update-models` | Fill `MODELS` in `settings.py` with the models downloaded on the server. |
| `--stats` | Show performance metrics after each answer, and a summary table when several models are run. |
| `--nothinking` | Hide the model's thinking and only show the answer. The model still thinks, so it isn't faster. |
| `--think=off` | Turn thinking off in the model itself, so answers come faster (possibly worse). `on` turns it on, and `low`/`medium`/`high` set how much it thinks on models that support it. Without this switch the model decides. Models that can't think give an error with `on`. |
| `--system "TEXT"` | Give the model a role or instructions, e.g. `--system "You are a security analyst. Answer briefly."` |

Switches can be combined:

```bash
python3 query.py --model=1-3 --stats --nothinking What is 6+6
```

### Piping in files

Text piped in is added after the question, so you can ask about files and command output:

```bash
cat script.py | python3 query.py --model=4 Review this code
python3 query.py --model=2 Summarize the errors in this log < /var/log/syslog
```

If there is no question on the command line, the piped text is used as the question. For big files, raise `num_ctx` in `settings.py`.

If the server can't be reached, the script stops after 5 seconds with a message instead of waiting.
