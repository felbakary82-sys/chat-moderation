"""Area B - minimum edit distance (Levenshtein) lexicon matcher.

This is our rule-based baseline: a message is offensive if one of its words is
within a small edit distance of a word in the offensive lexicon. Edit distance
catches misspellings the regexes missed ("idoit", "stupd", "id*ot").
"""
from functools import lru_cache
from pathlib import Path

from .normalize import normalize, tokenize

LEXICON_PATH = Path(__file__).resolve().parent / "offensive_lexicon.txt"
WILDCARD = "*"   # "id*ot": a star may stand for any letter (substitution cost 0)


def load_lexicon(path=LEXICON_PATH):
    words = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip().lower()
        if line and not line.startswith("#"):
            words.append(line)
    return words


def edit_distance(a: str, b: str) -> int:
    """Minimum edit distance with insert = delete = substitute = 1 (dynamic programming).

    D[i][j] = distance between the first i chars of a and the first j chars of b.
    """
    n, m = len(a), len(b)
    prev = list(range(m + 1))                 # row for i = 0
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            same = a[i - 1] == b[j - 1] or a[i - 1] == WILDCARD or b[j - 1] == WILDCARD
            cur[j] = min(prev[j] + 1,                      # deletion
                         cur[j - 1] + 1,                   # insertion
                         prev[j - 1] + (0 if same else 1)) # substitution / match
        prev = cur
    return prev[m]


def max_distance(word: str) -> int:
    """Allowed typos grow with word length. Words of 4 letters or less must match
    exactly, otherwise innocent words match ("such" -> "suck", "that" -> "twat")."""
    if len(word) <= 4:
        return 0
    if len(word) <= 7:
        return 1
    return 2


class LexiconMatcher:
    def __init__(self, words=None):
        words = words if words is not None else load_lexicon()
        self.roots = [w.rstrip("*") for w in words if w.endswith("*")]
        words = [w.rstrip("*") for w in words]
        self.words = [w for w in words if " " not in w]
        self.phrases = [w for w in words if " " in w]

    @lru_cache(maxsize=50_000)
    def match_token(self, token: str):
        """Return (lexicon_word, distance) for the closest match, or None."""
        token = token.strip("'")
        if len(token) < 3 or token == "@user":
            return None
        for root in self.roots:                # "youidiot" contains "idiot"
            if root in token and len(token) > len(root):
                return (root, 0)
        best = None
        for w in self.words:
            limit = max_distance(w)
            if abs(len(token) - len(w)) > limit:
                continue
            d = edit_distance(token, w)
            if d <= limit and (best is None or d < best[1]):
                best = (w, d)
                if d == 0:
                    break
        return best

    def find(self, normalized_text: str):
        """All matches in an already-normalized message: list of (token, word, distance)."""
        hits = []
        for tok in tokenize(normalized_text):
            m = self.match_token(tok)
            if m:
                hits.append((tok, m[0], m[1]))
        for p in self.phrases:
            if p in normalized_text:
                hits.append((p, p, 0))
        return hits

    def mask(self, original_text: str) -> str:
        """Hide offensive words in the ORIGINAL text: 'you idiot' -> 'you i****'."""
        out = []
        for word in original_text.split(" "):
            norm = normalize(word)
            if any(self.match_token(t) for t in tokenize(norm)):
                core = word.strip(".,!?;:\"'")
                out.append(word.replace(core, core[:1] + "*" * (len(core) - 1)) if core else word)
            else:
                out.append(word)
        return " ".join(out)
