"""The application: a live chat where every message is moderated.

  python app.py            -> open http://127.0.0.1:7860
  python app.py --share    -> also creates a public link anyone can open

Default model = quantized BERT (INT8). Each message is labeled clean / offensive
with a confidence; offensive messages are masked; the side panel logs the
response time of every message and lets you switch models.
"""
import argparse
import html
import time

import gradio as gr
import pandas as pd

import config
from moderation.lexicon import LexiconMatcher
from moderation.models import load_models

print("Loading models...")
MODELS = load_models(config.MODELS_DIR)
if not MODELS:
    raise SystemExit("No models found in models/. Run download_models.py or train_all.py first.")
DEFAULT = "BERT INT8 (quantized)" if "BERT INT8 (quantized)" in MODELS else list(MODELS)[0]
MATCHER = LexiconMatcher()
BUDGET = config.LATENCY_BUDGET_MS
LOG_COLUMNS = ["#", "Model", "Label", "Confidence", "Latency (ms)", f"<= {BUDGET} ms"]

CSS = """
.chatbox {height: 430px; overflow-y: auto; padding: 10px; border-radius: 10px;
          background: #f4f5f7; color: #111; font-family: sans-serif;}
.msg {background: white; border-radius: 10px; padding: 8px 12px; margin: 8px 0;
      box-shadow: 0 1px 2px rgba(0,0,0,.08); color: #111;}
.tag {font-size: 12px; font-weight: bold; padding: 2px 8px; border-radius: 8px; color: white;}
.clean {background: #2e9d5b;} .offensive {background: #d64545;}
.meta {font-size: 11px; color: #666; margin-top: 4px;}
details {font-size: 11px; color: #888;}
"""


def timed_predict(model, text):
    start = time.perf_counter()
    label, conf = model.predict(text)
    return label, conf, (time.perf_counter() - start) * 1000


def mask(text):
    masked = MATCHER.mask(text)
    # a model can flag a message with no lexicon word in it -> hide the whole message
    return masked if masked != text else "[message hidden - flagged as offensive]"


def render_chat(history):
    if not history:
        return "<div class='chatbox'><i>No messages yet. Type something below.</i></div>"
    parts = []
    for h in history:
        label = config.LABELS[h["label"]]
        if h["label"] == 1:
            body = (f"{html.escape(h['masked'])}"
                    f"<details><summary>show original</summary>{html.escape(h['text'])}</details>")
        else:
            body = html.escape(h["text"])
        parts.append(
            f"<div class='msg'><span class='tag {label}'>{label.upper()} {h['conf']:.0%}</span> "
            f"{body}<div class='meta'>{html.escape(h['model'])} - {h['ms']:.1f} ms</div></div>")
    return "<div class='chatbox'>" + "".join(reversed(parts)) + "</div>"


def send(text, model_name, compare, history, log):
    text = (text or "").strip()
    if not text:
        return "", render_chat(history), history, log, pd.DataFrame(log, columns=LOG_COLUMNS), gr.update()
    label, conf, ms = timed_predict(MODELS[model_name], text)
    history = history + [{"text": text, "masked": mask(text) if label else text,
                          "label": label, "conf": conf, "model": model_name, "ms": ms}]
    log = log + [[len(log) + 1, model_name, config.LABELS[label], round(conf, 3), round(ms, 2),
                  "yes" if ms <= BUDGET else "no"]]

    compare_df = gr.update()
    if compare:
        rows = []
        for name, m in MODELS.items():
            lb, cf, t = timed_predict(m, text)
            rows.append([name, config.LABELS[lb], round(cf, 3), round(t, 2)])
        compare_df = pd.DataFrame(rows, columns=["Model", "Label", "Confidence", "Latency (ms)"])
    log_df = pd.DataFrame(log, columns=LOG_COLUMNS)
    return "", render_chat(history), history, log, log_df, compare_df


def clear():
    return render_chat([]), [], [], pd.DataFrame([], columns=LOG_COLUMNS)


with gr.Blocks(title="Real-Time Chat Moderation", css=CSS) as demo:
    gr.Markdown("# Real-Time Chat Moderation\nEvery message is classified **clean / offensive** "
                f"on CPU. Latency budget: **{BUDGET} ms** per message.")
    history_state, log_state = gr.State([]), gr.State([])
    with gr.Row():
        with gr.Column(scale=3):
            chat = gr.HTML(render_chat([]))
            with gr.Row():
                box = gr.Textbox(placeholder="Type a chat message and press Enter...",
                                 show_label=False, scale=5)
                send_btn = gr.Button("Send", variant="primary", scale=1)
            gr.Examples(["gg everyone, that was a fun match!",
                         "you are such an idiot lol",
                         "y o u a r e a n i d i o t",
                         "id1ot team, uninstall the game",
                         "@USER this player is trash and so are you"],
                        inputs=box)
        with gr.Column(scale=2):
            model_dd = gr.Dropdown(list(MODELS), value=DEFAULT, label="Model used for the chat")
            compare_cb = gr.Checkbox(value=True, label="Also compare all models on each message")
            log_table = gr.Dataframe(pd.DataFrame([], columns=LOG_COLUMNS),
                                     label="Response time log", interactive=False)
            compare_table = gr.Dataframe(label="All models on the last message", interactive=False)
            clear_btn = gr.Button("Clear chat")

    outputs = [box, chat, history_state, log_state, log_table, compare_table]
    inputs = [box, model_dd, compare_cb, history_state, log_state]
    box.submit(send, inputs, outputs)
    send_btn.click(send, inputs, outputs)
    clear_btn.click(clear, None, [chat, history_state, log_state, log_table])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    demo.launch(share=args.share, server_port=args.port)
