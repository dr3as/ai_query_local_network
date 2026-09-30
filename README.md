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

## Conversations with memory (`query_history.py`)

`query_history.py` lets you have a conversation with a model without sending the whole chat history with every question. Instead, each conversation keeps a short **memory**: a summary of what has been said so far. For every question:

1. The memory and the conversation's system prompt are sent along with your question.
2. The answer is streamed to the screen, like in `query.py`.
3. The model rewrites the memory to include the new question and answer, and it is saved.

This keeps the prompt small no matter how long the conversation gets. The trade-off is that details the summary drops are gone for the model.

### Usage

Start a new conversation by asking a question without `--convo`. It is saved as `convos/convo_001.json`, `convo_002.json` and so on:

```bash
python3 query_history.py --model=5 --system "You are a running coach. Answer briefly." I'm training for a half marathon in May
```

Continue it with `--convo` and the conversation's number:

```bash
python3 query_history.py --convo=1 How long should my long runs be
```

Find the right number with `--list`, which shows each conversation's number, last update, turns, model and first question:

```bash
python3 query_history.py --list
```

| Switch | What it does |
|---|---|
| `--convo=N` | Continue conversation `N`, or a convo file given by its path. Without it, a new conversation is started. |
| `--list` | List the saved conversations. |
| `--model=N` | Model number or name. A new conversation uses the first model in `MODELS` unless you pick one. Using `--model` on an existing conversation switches it to that model from then on. |
| `--system "TEXT"` | System prompt, saved in the conversation and used for every later question. Giving it again replaces it. |
| `--stats`, `--nothinking`, `--think` | Same as in `query.py`. |

Piping in text works the same as in `query.py`.

### The convo file

Each conversation is a JSON file in `convos/`, which is ignored by git:

| Field | What it holds |
|---|---|
| `id`, `title` | The conversation's number, and the start of the first question. |
| `model`, `system` | The model and system prompt used for the next question. |
| `summary` | The memory that is sent with each question. You can edit it by hand. |
| `turns`, `created`, `updated` | Number of questions asked, and when. |
| `log` | Every question and answer in full, for you to read back. It is **not** sent to the model. |

The memory's max length is set with `HISTORY_SUMMARY_WORDS` in `settings.py` (default 300 words). A longer memory remembers more details, but uses more of `num_ctx` and makes each question a bit slower.
