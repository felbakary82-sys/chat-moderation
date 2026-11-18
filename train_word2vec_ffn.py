"""Step 3 - our own word2vec + feedforward network (Area A).

  a) train word2vec (skip-gram, negative sampling) on the TRAINING tweets only
  b) represent each message as the average of its word vectors
  c) train a feedforward network on those vectors with SGD + backpropagation
"""
import random
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

import config
from moderation.neural import FeedForward, SkipGram, average_vectors
from moderation.normalize import normalize, tokenize


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_pairs(sentences, stoi, window, rng):
    """(center, context) id pairs with a random window size, like the original word2vec."""
    centers, contexts = [], []
    for toks in sentences:
        ids = [stoi[t] for t in toks if t in stoi]
        for i, c in enumerate(ids):
            w = rng.integers(1, window + 1)
            for j in range(max(0, i - w), min(len(ids), i + w + 1)):
                if j != i:
                    centers.append(c)
                    contexts.append(ids[j])
    return torch.tensor(centers), torch.tensor(contexts)


def train_word2vec(sentences):
    counts = Counter(t for s in sentences for t in s)
    vocab = [w for w, c in counts.most_common() if c >= config.W2V_MIN_COUNT]
    stoi = {w: i for i, w in enumerate(vocab)}
    print(f"word2vec vocabulary: {len(vocab)} words")

    # negative samples are drawn from unigram^0.75 (word2vec paper)
    freq = torch.tensor([counts[w] for w in vocab], dtype=torch.float) ** 0.75
    neg_dist = freq / freq.sum()

    model = SkipGram(len(vocab), config.W2V_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    rng = np.random.default_rng(config.SEED)
    batch = 1024
    for epoch in range(config.W2V_EPOCHS):
        centers, contexts = build_pairs(sentences, stoi, config.W2V_WINDOW, rng)
        perm = torch.randperm(len(centers))
        total, t0 = 0.0, time.time()
        for k in range(0, len(perm), batch):
            idx = perm[k:k + batch]
            neg = torch.multinomial(neg_dist, len(idx) * config.W2V_NEGATIVES, replacement=True)
            loss = model(centers[idx], contexts[idx], neg.view(len(idx), -1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        print(f"  w2v epoch {epoch + 1}/{config.W2V_EPOCHS}  pairs={len(perm)}  "
              f"loss={total / len(perm):.4f}  ({time.time() - t0:.0f}s)")
    return vocab, stoi, model.in_emb.weight.detach().clone()


def show_neighbours(words, stoi, vocab, emb):
    norm = emb / emb.norm(dim=1, keepdim=True).clamp_min(1e-8)
    for w in words:
        if w in stoi:
            sims = norm @ norm[stoi[w]]
            best = sims.topk(6).indices.tolist()[1:]
            print(f"  nearest to '{w}': {', '.join(vocab[i] for i in best)}")


def main():
    set_seed(config.SEED)
    train = pd.read_csv(config.DATA_DIR / "train.csv")
    val = pd.read_csv(config.DATA_DIR / "validation.csv")
    train_toks = [tokenize(normalize(t)) for t in train["text"]]
    val_toks = [tokenize(normalize(t)) for t in val["text"]]

    # a) word2vec
    vocab, stoi, emb = train_word2vec(train_toks)
    show_neighbours(["stupid", "love", "trump", "liberals", "gun"], stoi, vocab, emb)

    # b) average vectors
    x_train = average_vectors(train_toks, stoi, emb)
    x_val = average_vectors(val_toks, stoi, emb)
    y_train = torch.tensor(train["label"].values)

    # c) feedforward network, SGD + backpropagation, class weights for imbalance
    ffn = FeedForward(config.W2V_DIM, config.FFN_HIDDEN)
    counts = torch.bincount(y_train, minlength=2).float()
    loss_fn = nn.CrossEntropyLoss(weight=counts.sum() / (2 * counts))
    opt = torch.optim.SGD(ffn.parameters(), lr=0.05, momentum=0.9)
    best_f1, best_state = -1, None
    for epoch in range(config.FFN_EPOCHS):
        ffn.train()
        perm = torch.randperm(len(x_train))
        for k in range(0, len(perm), 32):
            idx = perm[k:k + 32]
            loss = loss_fn(ffn(x_train[idx]), y_train[idx])
            opt.zero_grad()
            loss.backward()          # backpropagation
            opt.step()               # SGD update
        ffn.eval()
        with torch.no_grad():
            pred = ffn(x_val).argmax(-1).numpy()
        f1 = f1_score(val["label"], pred, average="macro")
        if f1 > best_f1:
            best_f1, best_state = f1, {k: v.clone() for k, v in ffn.state_dict().items()}
        if (epoch + 1) % 5 == 0:
            print(f"  FFN epoch {epoch + 1:2d}  loss={loss.item():.4f}  val macro-F1={f1:.4f}")

    print(f"Best validation macro-F1: {best_f1:.4f}")
    config.MODELS_DIR.mkdir(exist_ok=True)
    torch.save({"vocab": vocab, "emb": emb, "ffn": best_state,
                "dim": config.W2V_DIM, "hidden": config.FFN_HIDDEN},
               config.MODELS_DIR / "w2v_ffn.pt")
    print("Saved models/w2v_ffn.pt")


if __name__ == "__main__":
    main()
