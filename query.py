import argparse
import re
import sys
import time
from pathlib import Path

from ollama import Client
import settings

SETTINGS_FILE = Path(__file__).with_name("settings.py")


def update_models(client):
    """Henter modellene som er lastet ned på serveren og skriver dem til MODELS i settings.py."""
    models = sorted(m.model for m in client.list().models)
    block = "MODELS = [\n" + "".join(f'    "{m}",\n' for m in models) + "]\n"

    text = SETTINGS_FILE.read_text()
    pattern = re.compile(r"^MODELS\s*=\s*\[.*?\]\n?", re.MULTILINE | re.DOTALL)
    if pattern.search(text):
        text = pattern.sub(lambda _: block, text, count=1)
    else:
        text = text.rstrip("\n") + "\n\n" + block
    SETTINGS_FILE.write_text(text)

    print(f"Oppdaterte {SETTINGS_FILE.name} med {len(models)} modeller:")
    for m in models:
        print(f"  • {m}")


parser = argparse.ArgumentParser(description="Test modeller på en Ollama-server i lokalnettet.")
parser.add_argument("--update-models", action="store_true",
                    help="hent modellene som er lastet ned på serveren og skriv dem til settings.py")
parser.add_argument("--list-models", action="store_true",
                    help="vis modellene i settings.py med nummer")
parser.add_argument("--model", type=int, metavar="N",
                    help="kjør bare modell nummer N (se --list-models)")
args = parser.parse_args()

SERVER_IP = f"http://{settings.IP}:11434"
client = Client(host=SERVER_IP, timeout=600.0)

if args.update_models:
    update_models(client)
    sys.exit(0)

# Liste over modellene du vil sammenligne/teste efter hverandre (fra settings.py)
MODELS = getattr(settings, "MODELS", [])
if not MODELS:
    sys.exit("Ingen modeller i settings.py. Kjør med --update-models eller legg dem inn i MODELS.")

if args.list_models:
    for i, model_name in enumerate(MODELS, start=1):
        print(f"{i:>3}. {model_name}")
    sys.exit(0)

if args.model is not None:
    if not 1 <= args.model <= len(MODELS):
        sys.exit(f"Ugyldig modellnummer {args.model}. Velg 1-{len(MODELS)} (se --list-models).")
    MODELS = [MODELS[args.model - 1]]

prompt = "what is 1 + 1?"

print(f"Starter test mot {len(MODELS)} modeller på {SERVER_IP}...\n")

for model_name in MODELS:
    print("\n" + "=" * 70)
    print(f"Kjører modell: {model_name}")
    print("=" * 70)

    start_wall_time = time.perf_counter()

    try:
        response = client.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            options={
                "num_predict": -1,  # Ubegrenset tokens (avbrytes ikke midt i tenkingen)
                "num_ctx": 8192,     # Gir god plass til både tenking og svar
                "temperature": 0.3   # Litt lavere temp for mer strukturert resonnering
            },
            keep_alive=0            # Tømmer VRAM umiddelbart etter at svaret er ferdig
        )

        end_wall_time = time.perf_counter()
        total_wall_time = end_wall_time - start_wall_time

        # Hent ut tenkeprosessen (CoT) og det endelige svaret
        thinking = getattr(response.message, 'thinking', None)
        answer = response.message.content

        # Print ut tenkeprosessen hvis modellen genererte det
        if thinking:
            print("\n--- TENKEPROSESS (CHAIN OF THOUGHT) ---")
            print(thinking.strip())
            print("-" * 70)

        print("\n--- ENDELIG SVAR ---")
        print(answer.strip() if answer else "[Ingen sluttrespons ble levert]")
        print("-" * 70)

        # Hent ut Ollama sine interne beregninger (konverter fra nanosekunder til sekunder)
        load_sec = response.get('load_duration', 0) / 1e9
        prompt_eval_sec = response.get('prompt_eval_duration', 0) / 1e9
        eval_sec = response.get('eval_duration', 0) / 1e9

        prompt_tokens = response.get('prompt_eval_count', 0)
        eval_tokens = response.get('eval_count', 0)

        # Beregn tokens per sekund for responsgenereringen
        tok_per_sec = (eval_tokens / eval_sec) if eval_sec > 0 else 0.0

        print("\nPERFORMANCE METRICS:")
        print(f"  • Generation Speed  : {tok_per_sec:.2f} tokens/sec")
        print(f"  • Genererte tokens  : {eval_tokens} tokens ({eval_sec:.2f}s)")
        print(f"  • Modellinnlasting   : {load_sec:.2f}s")
        print(f"  • Prompt evaluering : {prompt_eval_sec:.2f}s ({prompt_tokens} tokens)")
        print(f"  • Total tid (Wall)  : {total_wall_time:.2f}s")

    except Exception as e:
        print(f"\nFeil under kjøring av {model_name}: {e}")

    # Kort pause for å la GPU/Ollama deallokere VRAM ordentlig før neste modell lastes inn
    time.sleep(1)
