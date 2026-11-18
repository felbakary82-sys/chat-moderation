"""Area A - our own word2vec (skip-gram with negative sampling) and a feedforward network.

Everything here is plain PyTorch so you can see every step:
  word2vec : learns a 100-number vector per word from the training tweets
  FFN      : average the word vectors of a message -> hidden layer -> 2 classes
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGram(nn.Module):
    """word2vec skip-gram: make a word's vector close to its real neighbours and
    far from random 'negative' words."""

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.in_emb = nn.Embedding(vocab_size, dim)    # the vectors we keep
        self.out_emb = nn.Embedding(vocab_size, dim)   # context vectors (thrown away)
        nn.init.uniform_(self.in_emb.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.out_emb.weight)

    def forward(self, center, context, negatives):
        c = self.in_emb(center)                           # (B, D)
        pos = self.out_emb(context)                       # (B, D)
        neg = self.out_emb(negatives)                     # (B, K, D)
        pos_score = (c * pos).sum(-1)                     # dot product
        neg_score = torch.bmm(neg, c.unsqueeze(-1)).squeeze(-1)
        loss = -F.logsigmoid(pos_score) - F.logsigmoid(-neg_score).sum(-1)
        return loss.mean()


class FeedForward(nn.Module):
    """Averaged word2vec vector (100) -> hidden layer (ReLU, dropout) -> 2 logits."""

    def __init__(self, dim: int, hidden: int, n_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def average_vectors(token_lists, stoi, emb):
    """Mean of the word vectors of each message (unknown words are skipped)."""
    out = torch.zeros(len(token_lists), emb.shape[1])
    for i, toks in enumerate(token_lists):
        ids = [stoi[t] for t in toks if t in stoi]
        if ids:
            out[i] = emb[ids].mean(0)
    return out
