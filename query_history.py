"""Have a conversation with a model without sending the whole chat history every time.

Each conversation is stored in convos/convo_NNN.json. Instead of the full history, the
model gets the last few exchanges word for word, plus a short summary ("memory") of
everything before them. When an exchange gets too old to be sent in full, the model
rewrites the memory to include it. The full conversation is kept in the file's log, so the
memory can be rebuilt from it (--resummarize), or sent instead of the memory (--full).
"""
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import settings
from query import (CONNECT_ERRORS, KEEP_ALIVE, OPTIONS, THINK_VALUES, load_models, make_client,
                   print_stats, read_prompt, run_model, select_models, server_unreachable,
                   SERVER_URL)

CONVO_DIR = Path(__file__).with_name("convos")
SUMMARY_WORDS = getattr(settings, "HISTORY_SUMMARY_WORDS", 300)
RECENT_TURNS = getattr(settings, "HISTORY_RECENT_TURNS", 2)

SUMMARY_SYSTEM = "You keep the memory of an ongoing conversation between a user and an AI assistant."

SUMMARY_PROMPT = """CURRENT MEMORY:
{summary}

NEW EXCHANGES:
{exchanges}

Rewrite the memory so it also covers the new exchanges. The assistant will only see this
memory, not these exchanges, so it must hold everything needed to continue.
Rules:
- Keep facts, names, numbers, decisions, the user's preferences and open questions.
- If the new exchanges change something already in the memory, update that point
  instead of adding a new one, so the memory never contradicts itself.
- Keep code, commands and exact values only if they are likely to be needed again.
- Drop greetings, filler, repetition and explanations that don't need repeating.
- Write short bullet points grouped by topic.
- Stay under {words} words. If space runs out, drop the least important old details first.
- Write only the memory, with no introduction or closing remarks."""


def convo_path(n):
    return CONVO_DIR / f"convo_{n:03d}.json"


def find_convo(spec):
    """Turn a --convo value (a number or a file path) into the path of an existing conversation."""
    path = convo_path(int(spec)) if spec.isdigit() else Path(spec)
    if not path.exists():
        sys.exit(f"Conversation {spec} not found ({path}). See --list.")
    return path


def new_convo(model, system):
    numbers = [int(m[1]) for p in CONVO_DIR.glob("convo_*.json")
               if (m := re.fullmatch(r"convo_(\d+)\.json", p.name))]
    n = max(numbers, default=0) + 1
    now = datetime.now().isoformat(timespec="seconds")
    convo = {"id": n, "created": now, "updated": now, "model": model, "system": system,
             "title": "", "turns": 0, "summary": "", "summarized_turns": 0, "log": []}
    return convo_path(n), convo


def save_convo(path, convo):
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(convo, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)  # Replace in one step, so a crash never leaves a half-written file


def list_convos():
    paths = sorted(CONVO_DIR.glob("convo_*.json"))
    if not paths:
        print("No conversations yet. Start one with: query_history.py Your question")
        return
    print(f"{'#':>4}  {'Updated':<16}  {'Turns':>5}  {'Model':<30}  Title")
    for p in paths:
        c = json.loads(p.read_text())
        print(f"{c['id']:>4}  {c['updated'][:16].replace('T', ' '):<16}  {c['turns']:>5}  "
              f"{c['model'][:30]:<30}  {c['title']}")


def strip_think_tags(text):
    """Some models write their thinking inside <think> tags in the answer itself."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def update_summary(client, convo, entries):
    """Ask the model to fold these log entries (question/answer pairs) into the conversation memory."""
    exchanges = "\n\n".join(f"User: {e['question']}\n\nAssistant: {strip_think_tags(e['answer'])}"
                            for e in entries)
    prompt = SUMMARY_PROMPT.format(
        summary=convo["summary"] or "(empty, these are the first exchanges)",
        exchanges=exchanges, words=SUMMARY_WORDS)
    # Summarizing needs no thinking, so turn it off where the model supports that
    can_think = "thinking" in (client.show(convo["model"]).capabilities or [])
    response = client.chat(
        model=convo["model"],
        messages=[{"role": "system", "content": SUMMARY_SYSTEM},
                  {"role": "user", "content": prompt}],
        think=False if can_think else None,
        options={**OPTIONS, "temperature": 0.1},
        keep_alive=KEEP_ALIVE,
    )
    return strip_think_tags(response.message.content or "")


def estimate_tokens(texts):
    """Rough token count (about 4 characters per token)."""
    return sum(len(t) for t in texts) // 4


def resummarize(client, convo):
    """Rebuild the memory from scratch from the full log. Returns True if it worked."""
    entries = convo["log"][:max(len(convo["log"]) - RECENT_TURNS, 0)]
    if not entries:
        print("Nothing to summarize yet: all exchanges are still sent word for word.")
        return True

    # Summarize in batches that fit comfortably in num_ctx, next to the memory itself
    budget = OPTIONS.get("num_ctx", 8192) // 2
    batches = [[]]
    for e in entries:
        if batches[-1] and estimate_tokens(
                [x["question"] + x["answer"] for x in batches[-1] + [e]]) > budget:
            batches.append([])
        batches[-1].append(e)

    old_summary = convo["summary"]
    convo["summary"] = ""
    for i, batch in enumerate(batches, start=1):
        print(f"Rebuilding the memory from {len(entries)} exchanges"
              + (f" (part {i}/{len(batches)})" if len(batches) > 1 else "") + "...",
              end="", flush=True)
        try:
            summary = update_summary(client, convo, batch)
        except CONNECT_ERRORS:
            server_unreachable()
        except Exception as e:
            summary = ""
            print(f" failed: {e}")
        if not summary:
            convo["summary"] = old_summary
            print(" Kept the old memory.")
            return False
        convo["summary"] = summary
        print(" done.")
    convo["summarized_turns"] = len(entries)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Talk to a model with a compact memory of the conversation saved in convos/.",
        epilog="Start a conversation: query_history.py What is a CVE. "
               "Continue it: query_history.py --convo=1 How are they scored")
    parser.add_argument("--convo", metavar="N",
                        help="continue conversation N (the number from --list) or a convo file path. "
                             "Without it, a new conversation is started")
    parser.add_argument("--list", action="store_true",
                        help="list the saved conversations")
    parser.add_argument("--model", metavar="N",
                        help="model number or name. New conversations default to the first model "
                             "in settings.py; continuing with --model switches the conversation to it")
    parser.add_argument("--full", action="store_true",
                        help="send the whole conversation word for word instead of the memory, "
                             "for this question only")
    parser.add_argument("--resummarize", action="store_true",
                        help="rebuild the memory from the full conversation. With a question, "
                             "this is done before asking it")
    parser.add_argument("--system", metavar="TEXT",
                        help="system prompt, saved in the conversation and used for every later question")
    parser.add_argument("--stats", action="store_true",
                        help="show performance metrics after the answer")
    parser.add_argument("--nothinking", action="store_true",
                        help="don't show the model's thinking, only the answer")
    parser.add_argument("--think", choices=THINK_VALUES,
                        help="turn the model's thinking on or off, or set how much it thinks")
    parser.add_argument("prompt", nargs="*", help="the question")
    args = parser.parse_args()

    if args.list:
        list_convos()
        return

    model = None
    if args.model is not None:
        selected = select_models(args.model, load_models())
        if len(selected) != 1:
            sys.exit("Choose exactly one model for a conversation.")
        model = selected[0]

    if args.resummarize and not args.convo:
        parser.error("--resummarize needs a conversation, e.g.: query_history.py --convo=1 --resummarize")

    question = read_prompt(args.prompt)
    if not question and not args.resummarize:
        parser.error("missing question, e.g.: query_history.py --convo=1 What is 6+6")

    if args.convo:
        path = find_convo(args.convo)
        convo = json.loads(path.read_text())
        if model:
            convo["model"] = model
        if args.system:
            convo["system"] = args.system
    else:
        path, convo = new_convo(model or load_models()[0], args.system)

    # Conversations from before summarized_turns existed had every turn in the memory
    convo.setdefault("summarized_turns", convo["turns"])

    client = make_client()

    if args.resummarize:
        if resummarize(client, convo):
            print(f"New memory:\n{convo['summary']}\n" if convo["summary"] else "", end="")
            save_convo(path, convo)
        if not question:
            return

    # With --full the whole log is sent word for word; otherwise the memory plus the
    # exchanges not yet in it. The memory goes after the user's own system prompt.
    system = convo["system"] or ""
    first_sent = 0 if args.full else convo["summarized_turns"]
    if convo["summary"] and not args.full:
        system += ("\n\n" if system else "") + (
            "Notes from earlier in this conversation (a summary of older exchanges; "
            "the most recent exchanges follow in full):\n" + convo["summary"])
    messages = [{"role": "system", "content": system}] if system else []
    for entry in convo["log"][first_sent:]:
        messages.append({"role": "user", "content": entry["question"]})
        messages.append({"role": "assistant", "content": strip_think_tags(entry["answer"])})
    messages.append({"role": "user", "content": question})

    print(f"Conversation {convo['id']} ({path.name}), turn {convo['turns'] + 1}, "
          f"model {convo['model']} on {SERVER_URL}"
          + (", with the full conversation" if args.full else ""))
    num_ctx = OPTIONS.get("num_ctx", 8192)
    if args.full and (tokens := estimate_tokens(m["content"] for m in messages)) > num_ctx * 0.8:
        print(f"Warning: the full conversation is about {tokens} tokens, close to or over num_ctx "
              f"({num_ctx}). The model may lose the start of it; raise num_ctx in settings.py.")

    try:
        stats, answer = run_model(client, convo["model"], messages,
                                  think=THINK_VALUES.get(args.think),
                                  show_thinking=not args.nothinking)
    except CONNECT_ERRORS:
        server_unreachable()
    except Exception as e:
        sys.exit(f"\nError while running {convo['model']}: {e}")
    if args.stats:
        print_stats(stats)

    now = datetime.now().isoformat(timespec="seconds")
    convo["turns"] += 1
    convo["updated"] = now
    convo["title"] = convo["title"] or question.splitlines()[0][:60]
    convo["log"].append({"time": now, "model": convo["model"], "question": question, "answer": answer})

    # Fold the exchanges that no longer fit among the recent ones into the memory
    to_summarize = convo["log"][convo["summarized_turns"]:len(convo["log"]) - RECENT_TURNS]
    if to_summarize:
        print("\nUpdating conversation memory...", end="", flush=True)
        try:
            summary = update_summary(client, convo, to_summarize)
        except Exception as e:
            summary = ""
            print(f" failed: {e}")
        if summary:
            convo["summary"] = summary
            convo["summarized_turns"] += len(to_summarize)
            print(" done.")
        else:
            print(" Kept the old memory. The next question will try again.")

    save_convo(path, convo)
    print(f"Saved. Continue with: query_history.py --convo={convo['id']} Your next question")


if __name__ == "__main__":
    main()
