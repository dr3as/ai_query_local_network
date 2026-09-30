import argparse
import re
import sys
import time
from pathlib import Path

from ollama import Client
import settings

SETTINGS_FILE = Path(__file__).with_name("settings.py")


def update_models(client):
    """Fetch the models downloaded on the server and write them to MODELS in settings.py."""
    models = sorted(m.model for m in client.list().models)
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


parser = argparse.ArgumentParser(description="Query models on an Ollama server on the local network.")
parser.add_argument("--update-models", action="store_true",
                    help="fetch the models downloaded on the server and write them to settings.py")
parser.add_argument("--list-models", action="store_true",
                    help="list the models in settings.py with their numbers")
parser.add_argument("--model", type=int, metavar="N",
                    help="run only model number N (see --list-models)")
parser.add_argument("--stats", action="store_true",
                    help="show performance metrics after each answer")
parser.add_argument("--nothinking", action="store_true",
                    help="don't show the model's thinking, only the answer")
parser.add_argument("prompt", nargs="*",
                    help="the question to send to the model(s)")
args = parser.parse_args()

SERVER_IP = f"http://{settings.IP}:11434"
client = Client(host=SERVER_IP, timeout=600.0)

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
    if not 1 <= args.model <= len(MODELS):
        sys.exit(f"Invalid model number {args.model}. Choose 1-{len(MODELS)} (see --list-models).")
    MODELS = [MODELS[args.model - 1]]

prompt = " ".join(args.prompt).strip()
if not prompt:
    parser.error("missing question, e.g.: query.py --model=1 What is 6+6")

print(f"Running {len(MODELS)} model(s) on {SERVER_IP}...\n")

for model_name in MODELS:
    print("\n" + "=" * 70)
    print(f"Model: {model_name}")
    print("=" * 70)

    start_wall_time = time.perf_counter()

    try:
        response = client.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            options={
                "num_predict": -1,  # Unlimited tokens (don't cut off mid-thinking)
                "num_ctx": 8192,     # Plenty of room for both thinking and answer
                "temperature": 0.3   # Slightly lower temp for more structured reasoning
            },
            keep_alive=0            # Free VRAM as soon as the answer is done
        )

        end_wall_time = time.perf_counter()
        total_wall_time = end_wall_time - start_wall_time

        # Get the thinking process (CoT) and the final answer
        thinking = getattr(response.message, 'thinking', None)
        answer = response.message.content

        # Print the thinking process if the model produced one
        if thinking and not args.nothinking:
            print("\n--- THINKING (CHAIN OF THOUGHT) ---")
            print(thinking.strip())
            print("-" * 70)

        print("\n--- ANSWER ---")
        print(answer.strip() if answer else "[No answer was returned]")
        print("-" * 70)

        if args.stats:
            # Get Ollama's internal timings (convert from nanoseconds to seconds)
            load_sec = response.get('load_duration', 0) / 1e9
            prompt_eval_sec = response.get('prompt_eval_duration', 0) / 1e9
            eval_sec = response.get('eval_duration', 0) / 1e9

            prompt_tokens = response.get('prompt_eval_count', 0)
            eval_tokens = response.get('eval_count', 0)

            # Tokens per second for generating the response
            tok_per_sec = (eval_tokens / eval_sec) if eval_sec > 0 else 0.0

            print("\nPERFORMANCE METRICS:")
            print(f"  • Generation speed  : {tok_per_sec:.2f} tokens/sec")
            print(f"  • Generated tokens  : {eval_tokens} tokens ({eval_sec:.2f}s)")
            print(f"  • Model load time   : {load_sec:.2f}s")
            print(f"  • Prompt evaluation : {prompt_eval_sec:.2f}s ({prompt_tokens} tokens)")
            print(f"  • Total time (wall) : {total_wall_time:.2f}s")

    except Exception as e:
        print(f"\nError while running {model_name}: {e}")

    # Short pause to let the GPU/Ollama free VRAM properly before the next model loads
    time.sleep(1)
