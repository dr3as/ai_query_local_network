import argparse
import re
import sys
import time
from pathlib import Path

import httpx
from ollama import Client
import settings

SETTINGS_FILE = Path(__file__).with_name("settings.py")

# Used when settings.py doesn't set them (see settings.example.py)
DEFAULT_OPTIONS = {
    "num_predict": -1,   # Unlimited tokens (don't cut off mid-thinking)
    "num_ctx": 8192,     # Plenty of room for both thinking and answer
    "temperature": 0.3,  # Slightly lower temp for more structured reasoning
}
DEFAULT_KEEP_ALIVE = 0   # Free VRAM as soon as the answer is done
DEFAULT_TIMEOUT = 600.0  # Seconds to wait for a model to answer

THINK_VALUES = {"on": True, "off": False, "low": "low", "medium": "medium", "high": "high"}


def server_unreachable():
    sys.exit(f"Could not connect to the Ollama server at {SERVER_URL}. "
             "Check that it is running and that IP in settings.py is correct.")


def update_models(client):
    """Fetch the models downloaded on the server and write them to MODELS in settings.py."""
    try:
        models = sorted(m.model for m in client.list().models)
    except (ConnectionError, httpx.ConnectError, httpx.ConnectTimeout):
        server_unreachable()
    block = "MODELS = [\n" + "".join(f'    "{m}",\n' for m in models) + "]\n"

    text = SETTINGS_FILE.read_text()
    pattern = re.compile(r"^MODELS\s*=\s*\[.*?\]\n?", re.MULTILINE | re.DOTALL)
    if pattern.search(text):
        text = pattern.sub(lambda _: block, text, count=1)
    else:
        text = text.rstrip("\n") + "\n\n" + block
    SETTINGS_FILE.write_text(text)

    print(f"Updated {SETTINGS_FILE.name} with {len(models)} models:")
    for m in models:
        print(f"  • {m}")


def select_models(spec, models):
    """Turn a --model value like "2", "1,3", "1-3" or "qwen3.5:9b" into a list of model names."""
    selected = []
    for part in (p.strip() for p in spec.split(",")):
        if not part:
            continue
        if part.isdigit():
            numbers = [int(part)]
        elif m := re.fullmatch(r"(\d+)-(\d+)", part):
            numbers = range(int(m[1]), int(m[2]) + 1)
        else:
            selected.append(part)  # A model name, used as is
            continue
        for n in numbers:
            if not 1 <= n <= len(models):
                sys.exit(f"Invalid model number {n}. Choose 1-{len(models)} (see --list-models).")
            selected.append(models[n - 1])
    return list(dict.fromkeys(selected))  # Drop duplicates, keep order


class SectionPrinter:
    """Prints streamed thinking/answer text under a header, without leading or trailing blank lines."""

    def __init__(self):
        self.section = None
        self.pending = ""  # Whitespace held back until more text arrives

    def write(self, section, text):
        if section != self.section:
            text = text.lstrip()
            if not text:
                return
            self.close()
            title = "THINKING (CHAIN OF THOUGHT)" if section == "thinking" else "ANSWER"
            print(f"\n--- {title} ---")
            self.section = section
        stripped = text.rstrip()
        if stripped:
            print(self.pending + stripped, end="", flush=True)
            self.pending = text[len(stripped):]
        else:
            self.pending += text

    def close(self):
        if self.section:
            print("\n" + "-" * 70)
        self.section = None
        self.pending = ""


def run_model(client, model_name, messages):
    """Stream one model's thinking and answer to the screen. Returns its stats."""
    start_wall_time = time.perf_counter()
    out = SectionPrinter()
    got_answer = False
    final = None

    for chunk in client.chat(
        model=model_name,
        messages=messages,
        stream=True,
        think=THINK,
        options=OPTIONS,
        keep_alive=KEEP_ALIVE,
    ):
        if chunk.message.thinking and not args.nothinking:
            out.write("thinking", chunk.message.thinking)
        if chunk.message.content:
            got_answer = got_answer or bool(chunk.message.content.strip())
            out.write("answer", chunk.message.content)
        if chunk.done:
            final = chunk

    if not got_answer:
        out.write("answer", "[No answer was returned]")
    out.close()

    # Ollama's internal timings, from the last chunk (nanoseconds to seconds)
    eval_sec = (final.eval_duration or 0) / 1e9 if final else 0.0
    eval_tokens = (final.eval_count or 0) if final else 0
    return {
        "tok_per_sec": (eval_tokens / eval_sec) if eval_sec > 0 else 0.0,
        "eval_tokens": eval_tokens,
        "eval_sec": eval_sec,
        "load_sec": (final.load_duration or 0) / 1e9 if final else 0.0,
        "prompt_eval_sec": (final.prompt_eval_duration or 0) / 1e9 if final else 0.0,
        "prompt_tokens": (final.prompt_eval_count or 0) if final else 0,
        "total_sec": time.perf_counter() - start_wall_time,
    }


def print_stats(s):
    print("\nPERFORMANCE METRICS:")
    print(f"  • Generation speed  : {s['tok_per_sec']:.2f} tokens/sec")
    print(f"  • Generated tokens  : {s['eval_tokens']} tokens ({s['eval_sec']:.2f}s)")
    print(f"  • Model load time   : {s['load_sec']:.2f}s")
    print(f"  • Prompt evaluation : {s['prompt_eval_sec']:.2f}s ({s['prompt_tokens']} tokens)")
    print(f"  • Total time (wall) : {s['total_sec']:.2f}s")


def print_summary(results):
    """Side-by-side comparison of all models that were run."""
    width = max(len("Model"), *(len(name) for name in results))
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<{width}}  {'Tokens/sec':>10}  {'Tokens':>7}  {'Load':>7}  {'Total':>7}")
    for name, s in results.items():
        if s is None:
            print(f"{name:<{width}}  {'error':>10}")
        else:
            print(f"{name:<{width}}  {s['tok_per_sec']:>10.2f}  {s['eval_tokens']:>7}"
                  f"  {s['load_sec']:>6.2f}s  {s['total_sec']:>6.2f}s")


parser = argparse.ArgumentParser(
    description="Query models on an Ollama server on the local network.",
    epilog="Text piped in on stdin is added to the question, e.g.: "
           "cat script.py | query.py --model=1 Review this code")
parser.add_argument("--update-models", action="store_true",
                    help="fetch the models downloaded on the server and write them to settings.py")
parser.add_argument("--list-models", action="store_true",
                    help="list the models in settings.py with their numbers")
parser.add_argument("--model", metavar="N",
                    help="run only these models: a number, a list or range of numbers, "
                         "or a model name (e.g. 2, 1,3,5, 1-3 or qwen3.5:9b)")
parser.add_argument("--stats", action="store_true",
                    help="show performance metrics after each answer, and a summary when running several models")
parser.add_argument("--nothinking", action="store_true",
                    help="don't show the model's thinking, only the answer")
parser.add_argument("--think", choices=THINK_VALUES,
                    help="turn the model's thinking on or off, or set how much it thinks "
                         "(low/medium/high, only some models). Default: the model decides")
parser.add_argument("--system", metavar="TEXT",
                    help='system prompt that gives the model a role or instructions, e.g. "You are a security analyst"')
parser.add_argument("prompt", nargs="*",
                    help="the question to send to the model(s)")
args = parser.parse_args()

OPTIONS = getattr(settings, "OPTIONS", DEFAULT_OPTIONS)
KEEP_ALIVE = getattr(settings, "KEEP_ALIVE", DEFAULT_KEEP_ALIVE)
TIMEOUT = getattr(settings, "TIMEOUT", DEFAULT_TIMEOUT)
THINK = THINK_VALUES.get(args.think)

SERVER_URL = f"http://{settings.IP}:11434"
# Fail fast if the server can't be reached, but give models plenty of time to answer
client = Client(host=SERVER_URL, timeout=httpx.Timeout(TIMEOUT, connect=5.0))

if args.update_models:
    update_models(client)
    sys.exit(0)

# The models to compare/test one after another (from settings.py)
MODELS = getattr(settings, "MODELS", [])
if not MODELS:
    sys.exit("No models in settings.py. Run with --update-models or add them to MODELS.")

if args.list_models:
    for i, model_name in enumerate(MODELS, start=1):
        print(f"{i:>3}. {model_name}")
    sys.exit(0)

if args.model is not None:
    MODELS = select_models(args.model, MODELS)
    if not MODELS:
        sys.exit("No models selected with --model.")

prompt = " ".join(args.prompt).strip()
if not sys.stdin.isatty():
    piped = sys.stdin.read().strip()
    if piped:
        prompt = f"{prompt}\n\n{piped}" if prompt else piped
if not prompt:
    parser.error("missing question, e.g.: query.py --model=1 What is 6+6")

messages = [{'role': 'user', 'content': prompt}]
if args.system:
    messages.insert(0, {'role': 'system', 'content': args.system})

print(f"Running {len(MODELS)} model(s) on {SERVER_URL}...\n")

results = {}
for model_name in MODELS:
    print("\n" + "=" * 70)
    print(f"Model: {model_name}")
    print("=" * 70)

    try:
        results[model_name] = run_model(client, model_name, messages)
        if args.stats:
            print_stats(results[model_name])
    except (ConnectionError, httpx.ConnectError, httpx.ConnectTimeout):
        server_unreachable()
    except Exception as e:
        results[model_name] = None
        print(f"\nError while running {model_name}: {e}")

    # Short pause to let the GPU/Ollama free VRAM properly before the next model loads
    if len(MODELS) > 1:
        time.sleep(1)

if args.stats and len(results) > 1:
    print_summary(results)
