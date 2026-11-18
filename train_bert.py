"""Step 4 - fine-tune BERT for offensive / clean classification (Area C).

bert-base-uncased (WordPiece tokenizer, 12 Transformer layers with self-attention)
+ a classification head on the [CLS] token. Plain PyTorch training loop.
Uses a GPU if there is one (Colab), otherwise CPU (slower, ~30-60 min).
"""
import argparse
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)

import config
from moderation.normalize import normalize


def encode(df, tokenizer):
    enc = tokenizer([normalize(t) for t in df["text"]], truncation=True, max_length=config.MAX_LEN)
    return [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i],
             "labels": int(df["label"].iloc[i])} for i in range(len(df))]


def evaluate(model, loader, device):
    model.eval()
    preds, gold = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds += model(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"]).logits.argmax(-1).tolist()
            gold += batch["labels"].tolist()
    return f1_score(gold, preds, average="macro")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=config.BERT_EPOCHS)
    ap.add_argument("--batch_size", type=int, default=config.BERT_BATCH)
    args = ap.parse_args()

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config.BERT_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.BERT_NAME, num_labels=2,
        id2label={0: "clean", 1: "offensive"}, label2id={"clean": 0, "offensive": 1}).to(device)

    train = pd.read_csv(config.DATA_DIR / "train.csv")
    val = pd.read_csv(config.DATA_DIR / "validation.csv")
    collate = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(encode(train, tokenizer), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate)
    val_loader = DataLoader(encode(val, tokenizer), batch_size=64, collate_fn=collate)

    counts = np.bincount(train["label"], minlength=2)
    weights = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=config.BERT_LR, weight_decay=0.01)
    steps = len(train_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    out_dir = config.MODELS_DIR / "bert_fp32"
    best = -1
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"]).logits
            loss = loss_fn(logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            if step % 50 == 0:
                print(f"  epoch {epoch + 1} step {step}/{len(train_loader)} loss={loss.item():.4f}")
        f1 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch + 1}: validation macro-F1 = {f1:.4f}  ({time.time() - t0:.0f}s)")
        if f1 > best:
            best = f1
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            print(f"  saved best model to {out_dir}")
    print(f"Best validation macro-F1: {best:.4f}")


if __name__ == "__main__":
    main()
