"""Step 6 - evaluate ALL models on the SAME held-out test set.

Writes:
  results/results.md      results table for the slides (metric, size, latency, hardware)
  results/results.csv     same numbers as CSV
  results/predictions.csv every model's prediction for every test message
  results/failures.csv    test messages the quantized BERT got wrong (for the slides)
"""
import random
import statistics
import time

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import config
from moderation.models import hardware_info, load_models


def latency_ms(model, texts):
    for t in texts[:10]:                      # warm-up runs are not counted
        model.predict(t)
    times = []
    for t in texts:                           # one message at a time, like a live chat
        start = time.perf_counter()
        model.predict(t)
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    return statistics.mean(times), times[int(0.95 * (len(times) - 1))]


def main():
    test = pd.read_csv(config.DATA_DIR / "test.csv")
    texts, gold = test["text"].tolist(), test["label"].tolist()
    print(f"Test set: {len(test)} messages ({sum(gold)} offensive)")
    print("Loading models...")
    models = load_models(config.MODELS_DIR)

    random.seed(config.SEED)
    lat_texts = random.sample(texts, min(config.LATENCY_RUNS, len(texts)))
    hw = hardware_info()
    print(f"Hardware: {hw}\n")

    rows, preds = [], {"text": texts, "gold": gold}
    for name, m in models.items():
        out = [m.predict(t) for t in texts]
        pred = [p for p, _ in out]
        preds[name] = pred
        preds[name + " conf"] = [round(c, 3) for _, c in out]
        mean_ms, p95_ms = latency_ms(m, lat_texts)
        rows.append({
            "Model": name,
            "Precision": m.precision,
            "Macro-F1": round(f1_score(gold, pred, average="macro"), 4),
            "Accuracy": round(accuracy_score(gold, pred), 4),
            "Size (MB)": round(m.size_mb, 3),
            "Latency mean (ms)": round(mean_ms, 2),
            "Latency p95 (ms)": round(p95_ms, 2),
            f"Within {config.LATENCY_BUDGET_MS} ms budget": "yes" if mean_ms <= config.LATENCY_BUDGET_MS else "no",
        })
        print(f"=== {name} ===")
        print(classification_report(gold, pred, target_names=config.LABELS, digits=4))
        print("Confusion matrix [rows = true clean/offensive, cols = predicted]:")
        print(confusion_matrix(gold, pred), "\n")

    table = pd.DataFrame(rows)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(config.RESULTS_DIR / "results.csv", index=False)
    pd.DataFrame(preds).to_csv(config.RESULTS_DIR / "predictions.csv", index=False)

    header = "| " + " | ".join(table.columns) + " |"
    sep = "|" + "---|" * len(table.columns)
    lines = ["| " + " | ".join(str(v) for v in r) + " |" for r in table.values]
    md = (f"# Results on the held-out test set ({len(test)} messages)\n\n"
          f"Latency: mean over {len(lat_texts)} single-message runs (after 10 warm-up runs).\n\n"
          f"Hardware: {hw}\n\n" + "\n".join([header, sep] + lines) + "\n")
    (config.RESULTS_DIR / "results.md").write_text(md, encoding="utf-8")

    int8 = "BERT INT8 (quantized)"
    if int8 in preds:
        df = pd.DataFrame(preds)
        wrong = df[df[int8] != df["gold"]][["text", "gold", int8, int8 + " conf"]]
        wrong.to_csv(config.RESULTS_DIR / "failures.csv", index=False)
        print(f"{len(wrong)} test messages misclassified by the quantized BERT -> results/failures.csv")

    print("\n" + table.to_string(index=False))
    print(f"\nSaved results to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
