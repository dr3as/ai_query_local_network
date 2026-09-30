import argparse
import re
import signal
import sys
import time
from pathlib import Path

import httpx
from ollama import Client
import settings

SETTINGS_FILE = Path(__file__).with_name("settings.py")

# Exit quietly when the output is piped to a command that stops reading, like head.
# Otherwise Python raises BrokenPipeError, which would look like a lost server connection.
if hasattr(signal, "SIGPIPE"):  # Not on Windows
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

# Used when settings.py doesn't set them (see settings.example.py)
DEFAULT_OPTIONS = {
    "num_predict": -1,   # Unlimited tokens (don't cut off mid-thinking)
    "num_ctx": 8192,     # Plenty of room for both thinking and answer
    "temperature": 0.3,  # Slightly lower temp for more structured reasoning
}
DEFAULT_KEEP_ALIVE = 0   # Free VRAM as soon as the answer is done
DEFAULT_TIMEOUT = 600.0  # Seconds to wait for a model to answer

THINK_VALUES = {"on": True, "off": False, "low": "low", "medium": "medium", "high": "high"}

OPTIONS = getattr(settings, "OPTIONS", DEFAULT_OPTIONS)
KEEP_ALIVE = getattr(settings, "KEEP_ALIVE", DEFAULT_KEEP_ALIVE)
TIMEOUT = getattr(settings, "TIMEOUT", DEFAULT_TIMEOUT)
SERVER_URL = f"http://{settings.IP}:11434"

# Errors that mean the server can't be reached at all
CONNECT_ERRORS = (ConnectionError, httpx.ConnectError, httpx.ConnectTimeout)


def make_client():
    # Fail fast if the server can't be reached, but give models plenty of time to answer
    return Client(host=SERVER_URL, timeout=httpx.Timeout(TIMEOUT, connect=5.0))


def load_models():
    """The models from settings.py, or exit if there are none."""
    models = getattr(settings, "MODELS", [])
    if not models:
        sys.exit("No models in settings.py. Run with --update-models or add them to MODELS.")
    return models


def read_prompt(words):
    """The question from the command line, with any text piped in on stdin added after it."""
    prompt = " ".join(words).strip()
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            prompt = f"{prompt}\n\n{piped}" if prompt else piped
    return prompt


def server_unreachable():
    sys.exit(f"Could not connect to the Ollama server at {SERVER_URL}. "
             "Check that it is running and that IP in settings.py is correct.")


def update_models(client):
    """Fetch the models downloaded on the server and write them to MODELS in settings.py."""
    try:
        models = sorted(m.model for m in client.list().models)
    except CONNECT_ERRORS:
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


def stream_chat(client, model_name, messages, think=None):
    """Yield ("thinking", text) and ("answer", text) pieces as the model writes them,
    and finally ("stats", dict) with Ollama's timings."""
    start_wall_time = time.perf_counter()
    final = None

    for chunk in client.chat(
        model=model_name,
        messages=messages,
        stream=True,
        think=think,
        options=OPTIONS,
        keep_alive=KEEP_ALIVE,
    ):
        if chunk.message.thinking:
            yield "thinking", chunk.message.thinking
        if chunk.message.content:
            yield "answer", chunk.message.content
        if chunk.done:
            final = chunk

    # Ollama's internal timings, from the last chunk (nanoseconds to seconds)
    eval_sec = (final.eval_duration or 0) / 1e9 if final else 0.0
    eval_tokens = (final.eval_count or 0) if final else 0
    yield "stats", {
        "tok_per_sec": (eval_tokens / eval_sec) if eval_sec > 0 else 0.0,
        "eval_tokens": eval_tokens,
        "eval_sec": eval_sec,
        "load_sec": (final.load_duration or 0) / 1e9 if final else 0.0,
        "prompt_eval_sec": (final.prompt_eval_duration or 0) / 1e9 if final else 0.0,
        "prompt_tokens": (final.prompt_eval_count or 0) if final else 0,
        "total_sec": time.perf_counter() - start_wall_time,
    }


def run_model(client, model_name, messages, think=None, show_thinking=True):
    """Stream one model's thinking and answer to the screen. Returns its stats and the answer."""
    out = SectionPrinter()
    answer = ""
    stats = None

    for kind, value in stream_chat(client, model_name, messages, think):
        if kind == "thinking" and show_thinking:
            out.write("thinking", value)
        elif kind == "answer":
            answer += value
            out.write("answer", value)
        elif kind == "stats":
            stats = value

    if not answer.strip():
        out.write("answer", "[No answer was returned]")
    out.close()
    return stats, answer.strip()


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


def main():
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

    client = make_client()

    if args.update_models:
        update_models(client)
        return

    # The models to compare/test one after another (from settings.py)
    models = load_models()

    if args.list_models:
        for i, model_name in enumerate(models, start=1):
            print(f"{i:>3}. {model_name}")
        return

    if args.model is not None:
        models = select_models(args.model, models)
        if not models:
            sys.exit("No models selected with --model.")

    prompt = read_prompt(args.prompt)
    if not prompt:
        parser.error("missing question, e.g.: query.py --model=1 What is 6+6")

    messages = [{'role': 'user', 'content': prompt}]
    if args.system:
        messages.insert(0, {'role': 'system', 'content': args.system})

    print(f"Running {len(models)} model(s) on {SERVER_URL}...\n")

    results = {}
    for model_name in models:
        print("\n" + "=" * 70)
        print(f"Model: {model_name}")
        print("=" * 70)

        try:
            results[model_name], _ = run_model(client, model_name, messages,
                                               think=THINK_VALUES.get(args.think),
                                               show_thinking=not args.nothinking)
            if args.stats:
                print_stats(results[model_name])
        except CONNECT_ERRORS:
            server_unreachable()
        except Exception as e:
            results[model_name] = None
            print(f"\nError while running {model_name}: {e}")

        # Short pause to let the GPU/Ollama free VRAM properly before the next model loads
        if len(models) > 1:
            time.sleep(1)

    if args.stats and len(results) > 1:
        print_summary(results)


if __name__ == "__main__":
    main()
