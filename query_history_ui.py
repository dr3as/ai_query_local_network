"""Full-screen terminal chat on top of query_history.py's conversations with memory.

Pick a conversation on the left (or create one), then chat like in a normal chat app.
Each question uses the same memory logic as query_history.py, and the convo files are shared.
"""
import time

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (Button, Collapsible, Footer, Input, Label, ListItem, ListView,
                             Markdown, Select, Static)
from textual.worker import get_current_worker

from query import CONNECT_ERRORS, SERVER_URL, load_models, make_client, stream_chat
from query_history import (RECENT_TURNS, all_convos, build_messages, context_warning, fold_memory,
                           new_convo, record_turn, resummarize, save_convo, strip_think_tags)

HELP = """Commands:
  /full QUESTION   ask with the whole conversation word for word instead of the memory
  /resummarize     rebuild the memory from the full conversation
  /memory          show the memory
  /system          show the system prompt
  /system TEXT     set the system prompt for the rest of the conversation
  /help            show this help"""


def model_options(extra=None):
    """(label, value) pairs for a model Select: the models from settings.py, plus `extra`
    if it isn't among them (e.g. a conversation's model that was removed from MODELS)."""
    models = list(load_models())
    if extra and extra not in models:
        models.append(extra)
    return [(m, m) for m in models]


class ConvoItem(ListItem):
    def __init__(self, path, convo):
        super().__init__(Label(f"{convo['id']:>3}  {convo['title'] or '(new conversation)'}"))
        self.path = path


class ConvoList(ListView):
    BINDINGS = [Binding("d,delete", "app.delete_convo", "Delete conversation")]


class AssistantMessage(Vertical):
    """One answer: the model's thinking (collapsible) and the answer as Markdown."""

    def __init__(self, model, answer=""):
        super().__init__(classes="assistant")
        self.border_title = model
        self.thinking = Collapsible(Static("", markup=False), title="Thinking", collapsed=False,
                                    classes="thinking")
        self.markdown = Markdown(answer)

    def compose(self) -> ComposeResult:
        yield self.thinking
        yield self.markdown

    async def show(self, thinking, answer, done=False):
        if thinking.strip():
            self.thinking.add_class("has-thinking")
            self.thinking.query_one(Static).update(thinking.strip())
            # Keep the thinking open while it's being written, fold it away once the answer starts
            self.thinking.collapsed = bool(answer.strip())
        text = strip_think_tags(answer) if done else answer
        await self.markdown.update(text or ("*[No answer was returned]*" if done else "…"))


class NewConvoScreen(ModalScreen):
    """Ask for the model and system prompt of a new conversation. Returns (model, system) or None."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, default_model):
        super().__init__()
        self.default_model = default_model

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("New conversation", classes="dialog-title")
            yield Label("Model")
            yield Select(model_options(self.default_model), value=self.default_model,
                         allow_blank=False, id="new-model")
            yield Label("System prompt (optional)")
            yield Input(placeholder="e.g. You are a running coach. Answer briefly.", id="new-system")
            with Horizontal(classes="buttons"):
                yield Button("Create", variant="primary", id="create")
                yield Button("Cancel", id="cancel")

    def on_mount(self):
        self.query_one("#new-system").focus()

    @on(Button.Pressed, "#create")
    @on(Input.Submitted, "#new-system")
    def create(self):
        system = self.query_one("#new-system", Input).value.strip() or None
        self.dismiss((self.query_one("#new-model", Select).value, system))

    @on(Button.Pressed, "#cancel")
    def action_cancel(self):
        self.dismiss(None)


class ConfirmScreen(ModalScreen):
    """A yes/no question. Returns True or False."""

    BINDINGS = [Binding("y", "answer(True)", "Yes"), Binding("n,escape", "answer(False)", "No")]

    def __init__(self, question):
        super().__init__()
        self.question = question

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(self.question)
            with Horizontal(classes="buttons"):
                yield Button("Yes (y)", variant="error", id="yes")
                yield Button("No (n)", id="no")

    def on_mount(self):
        self.query_one("#no").focus()

    def on_button_pressed(self, event):
        self.dismiss(event.button.id == "yes")

    def action_answer(self, yes):
        self.dismiss(yes)


class ChatApp(App):
    TITLE = "Ollama chat"
    CSS = """
    #sidebar { width: 34; border-right: solid $primary-darken-2; }
    #sidebar-title { padding: 0 1; text-style: bold; background: $primary-darken-2; width: 100%; }
    #convos { height: 1fr; }
    #topbar { height: 3; }
    #title { width: 1fr; padding: 1 1 0 1; text-style: bold; }
    #model { width: 44; }
    #status { color: $text-muted; padding: 0 1; height: 1; }
    #chat { height: 1fr; padding: 0 1; }
    .user { border: round $accent; margin: 1 0 0 8; padding: 0 1; height: auto;
            border-title-align: right; }
    .assistant { border: round $primary; margin: 1 8 0 0; padding: 0 1; height: auto; }
    .assistant Markdown { margin: 0; padding: 0; }
    .assistant Markdown > *:last-of-type { margin-bottom: 0; }
    .thinking { display: none; color: $text-muted; margin: 0 0 1 0; padding: 0; border: none; }
    .thinking.has-thinking { display: block; }
    .info { color: $text-muted; margin: 1 0 0 0; }
    .error { color: $error; margin: 1 0 0 0; }
    Screen.hide-thinking .thinking { display: none; }
    NewConvoScreen, ConfirmScreen { align: center middle; }
    #dialog { width: 70; height: auto; border: thick $primary; background: $surface; padding: 1 2; }
    #dialog Label { margin: 1 0 0 0; }
    .dialog-title { text-style: bold; }
    .buttons { height: auto; margin: 1 0 0 0; }
    .buttons Button { margin: 0 2 0 0; }
    """
    BINDINGS = [
        Binding("ctrl+n", "new_convo", "New conversation", priority=True),
        Binding("ctrl+o", "focus_model", "Change model", priority=True),
        Binding("ctrl+t", "toggle_thinking", "Show/hide thinking", priority=True),
        Binding("escape", "focus_list", "Conversations"),
        Binding("ctrl+q", "quit", "Quit", priority=True),
    ]

    def __init__(self):
        super().__init__()
        self.client = make_client()
        self.path = None
        self.convo = None

    def compose(self) -> ComposeResult:
        with Horizontal():
            with Vertical(id="sidebar"):
                yield Label("Conversations", id="sidebar-title")
                yield ConvoList(id="convos")
            with Vertical():
                with Horizontal(id="topbar"):
                    yield Label("", id="title")
                    yield Select(model_options(), allow_blank=False, id="model")
                yield Static("", id="status")
                yield VerticalScroll(id="chat")
                yield Input(placeholder="Choose a conversation first", id="input", disabled=True)
        yield Footer()

    async def on_mount(self):
        self.sub_title = SERVER_URL
        self.query_one("#model").disabled = True
        await self.refresh_list()
        await self.show_info("Choose a conversation on the left, or press Ctrl+N for a new one.")
        self.query_one("#convos").focus()

    # ---- Conversation list -------------------------------------------------------------

    async def refresh_list(self):
        """Reload the conversation list, keeping the open conversation highlighted."""
        convos = all_convos()
        lv = self.query_one("#convos", ConvoList)
        await lv.clear()
        await lv.extend([ConvoItem(p, c) for p, c in reversed(convos)])  # Newest first
        paths = [p for p, _ in reversed(convos)]
        if self.path in paths:
            lv.index = paths.index(self.path)

    @on(ListView.Selected, "#convos")
    async def convo_selected(self, event):
        if event.item.path != self.path:
            await self.open_convo(event.item.path)
        self.query_one("#input").focus()

    async def open_convo(self, path):
        self.path = path
        self.convo = next(c for p, c in all_convos() if p == path)
        chat = self.query_one("#chat")
        await chat.remove_children()
        for entry in self.convo["log"]:
            await chat.mount(Static(entry["question"], markup=False, classes="user"))
            chat.children[-1].border_title = "You"
            await chat.mount(AssistantMessage(entry.get("model", self.convo["model"]),
                                              strip_think_tags(entry["answer"])))
        if not self.convo["log"]:
            await self.show_info("New conversation. Type your first message below. /help lists commands.")
        chat.scroll_end(animate=False)

        select = self.query_one("#model", Select)
        with select.prevent(Select.Changed):
            select.set_options(model_options(self.convo["model"]))
            select.value = self.convo["model"]
        select.disabled = False
        inp = self.query_one("#input", Input)
        inp.disabled = False
        inp.placeholder = "Type a message and press Enter  (/help for commands)"
        self.update_header()

    def update_header(self):
        c = self.convo
        self.query_one("#title", Label).update(f"#{c['id']}  {c['title'] or '(new conversation)'}")
        memory = (f"memory covers {c['summarized_turns']}" if c["summary"] else "no memory yet")
        hidden = "  ·  thinking hidden" if self.screen.has_class("hide-thinking") else ""
        self.query_one("#status", Static).update(
            f"{c['turns']} turns  ·  {memory}  ·  last {RECENT_TURNS} sent word for word{hidden}")

    # ---- Actions ---------------------------------------------------------------------

    def action_new_convo(self):
        default = self.convo["model"] if self.convo else load_models()[0]

        async def created(result):
            if result is None:
                return
            model, system = result
            path, convo = new_convo(model, system)
            save_convo(path, convo)
            await self.open_convo(path)
            await self.refresh_list()
            self.query_one("#input").focus()

        self.push_screen(NewConvoScreen(default), created)

    def action_delete_convo(self):
        item = self.query_one("#convos", ConvoList).highlighted_child
        if item is None:
            return
        convo = next(c for p, c in all_convos() if p == item.path)

        async def confirmed(yes):
            if not yes:
                return
            item.path.unlink()
            if item.path == self.path:
                self.path = self.convo = None
                await self.query_one("#chat").remove_children()
                self.query_one("#title", Label).update("")
                self.query_one("#status", Static).update("")
                self.query_one("#model").disabled = True
                inp = self.query_one("#input", Input)
                inp.disabled = True
                inp.placeholder = "Choose a conversation first"
                await self.show_info(f"Deleted conversation {convo['id']}.")
            await self.refresh_list()
            self.query_one("#convos").focus()

        title = convo["title"] or "(new conversation)"
        self.push_screen(ConfirmScreen(f"Delete conversation {convo['id']}: {title}?"), confirmed)

    def action_focus_model(self):
        select = self.query_one("#model", Select)
        if not select.disabled:
            select.focus()
            select.expanded = True

    def action_focus_list(self):
        self.query_one("#convos").focus()

    def action_toggle_thinking(self):
        self.screen.toggle_class("hide-thinking")
        if self.convo:
            self.update_header()

    @on(Select.Changed, "#model")
    async def model_changed(self, event):
        if self.convo and event.value != self.convo["model"]:
            self.convo["model"] = event.value
            save_convo(self.path, self.convo)
            await self.show_info(f"Switched to {event.value}. The memory and history are kept.")
            self.query_one("#input").focus()

    # ---- Chat ------------------------------------------------------------------------

    async def show_info(self, text, error=False):
        chat = self.query_one("#chat")
        await chat.mount(Static(text, markup=False, classes="error" if error else "info"))
        chat.scroll_end(animate=False)

    @on(Input.Submitted, "#input")
    async def submitted(self, event):
        text = event.value.strip()
        if not text or not self.convo:
            return
        event.input.value = ""
        command, _, rest = text.partition(" ")
        rest = rest.strip()

        if command == "/help":
            await self.show_info(HELP)
        elif command == "/memory":
            await self.show_info(self.convo["summary"] or
                                 "No memory yet: all exchanges are still sent word for word.")
        elif command == "/system":
            if rest:
                self.convo["system"] = rest
                save_convo(self.path, self.convo)
                await self.show_info(f"System prompt set to: {rest}")
            else:
                await self.show_info(f"System prompt: {self.convo['system'] or '(none)'}")
        elif command == "/resummarize":
            self.set_busy(True)
            self.run_resummarize()
        elif command == "/full":
            if not rest:
                await self.show_info("Usage: /full QUESTION", error=True)
            else:
                await self.ask(rest, full=True)
        elif command.startswith("/"):
            await self.show_info(f"Unknown command {command}. Type /help for the list.", error=True)
        else:
            await self.ask(text, full=False)

    def set_busy(self, busy):
        inp = self.query_one("#input", Input)
        inp.disabled = busy
        self.query_one("#model").disabled = busy
        if not busy:
            inp.focus()

    async def ask(self, question, full):
        chat = self.query_one("#chat")
        await chat.mount(Static(question, markup=False, classes="user"))
        chat.children[-1].border_title = "You" + (" (full conversation)" if full else "")
        messages = build_messages(self.convo, question, full=full)
        if full and (warning := context_warning(messages)):
            await self.show_info(f"Warning: {warning}", error=True)
        message = AssistantMessage(self.convo["model"])
        await chat.mount(message)
        chat.scroll_end(animate=False)
        self.set_busy(True)
        self.run_question(question, messages, message)

    @work(thread=True, exclusive=True, group="model")
    def run_question(self, question, messages, message):
        """Stream the answer into `message`, then save the turn and update the memory."""
        worker = get_current_worker()
        convo, path = self.convo, self.path
        thinking = answer = ""
        last_update = 0.0
        try:
            for kind, value in stream_chat(self.client, convo["model"], messages):
                if worker.is_cancelled:
                    return
                if kind == "thinking":
                    thinking += value
                elif kind == "answer":
                    answer += value
                # Redraw at most 10 times a second; redrawing Markdown on every piece is slow
                if time.monotonic() - last_update > 0.1:
                    last_update = time.monotonic()
                    self.call_from_thread(self.update_message, message, thinking, answer)
            self.call_from_thread(self.update_message, message, thinking, answer, True)
        except CONNECT_ERRORS:
            self.call_from_thread(self.finish_with_error, message,
                                  f"Could not connect to the Ollama server at {SERVER_URL}.")
            return
        except Exception as e:
            self.call_from_thread(self.finish_with_error, message, f"Error: {e}")
            return

        record_turn(convo, question, strip_think_tags(answer))
        save_convo(path, convo)
        self.call_from_thread(self.after_answer)
        try:
            if fold_memory(self.client, convo) is False:
                self.call_from_thread(self.show_info, "Could not update the memory. "
                                      "The next question will try again.", True)
        except Exception as e:
            self.call_from_thread(self.show_info, f"Could not update the memory ({e}). "
                                  "The next question will try again.", True)
        save_convo(path, convo)
        self.call_from_thread(self.done)

    async def update_message(self, message, thinking, answer, done=False):
        await message.show(thinking, answer, done)
        self.query_one("#chat").scroll_end(animate=False)

    async def finish_with_error(self, message, text):
        await message.remove()
        await self.show_info(text, error=True)
        self.set_busy(False)

    async def after_answer(self):
        self.query_one("#status", Static).update("Updating memory...")
        await self.refresh_list()

    def done(self):
        self.update_header()
        self.set_busy(False)

    @work(thread=True, exclusive=True, group="model")
    def run_resummarize(self):
        convo, path = self.convo, self.path
        report = lambda text: self.call_from_thread(self.show_info, text)
        try:
            if resummarize(self.client, convo, report):
                save_convo(path, convo)
                if convo["summary"]:
                    report(f"New memory:\n{convo['summary']}")
        except CONNECT_ERRORS:
            self.call_from_thread(self.show_info,
                                  f"Could not connect to the Ollama server at {SERVER_URL}.", True)
        self.call_from_thread(self.done)


if __name__ == "__main__":
    ChatApp().run()
