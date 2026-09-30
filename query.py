import time
from ollama import Client
from settings import IP


SERVER_IP = f"http://{IP}:11434"
client = Client(host=SERVER_IP, timeout=600.0)

# Liste over modellene du vil sammenligne/teste efter hverandre
MODELS = [
    "qwen3.5:9b",
    #"",
    #"",
    #""
]

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
