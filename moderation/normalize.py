"""Area B - regex normalization of noisy chat text.

Gamers hide bad words with tricks: leetspeak (id1ot), repeated letters (stuuuupid),
spacing (s t u p i d), symbols (id*ot), Turkish characters. We undo these with regular
expressions so every model sees cleaner text.
"""
import re
import unicodedata

TURKISH_MAP = str.maketrans({
    "ı": "i", "İ": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g",
    "ç": "c", "Ç": "c", "ö": "o", "Ö": "o", "ü": "u", "Ü": "u",
})
LEET_MAP = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t",
            "@": "a", "$": "s", "!": "i"}

URL_RE = re.compile(r"(https?://\S+|www\.\S+|\burl\b)")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
# leet characters with a letter on both sides: id1ot, l0ser, n00b
LEET_INSIDE_RE = re.compile(r"(?<=[a-z])[013457@$!]+(?=[a-z])")
# $ or 5 at the start of a word followed by letters: $hit, 5hit
LEET_START_RE = re.compile(r"(?<![a-z0-9])[$5](?=[a-z]{2})")
# 4+ single letters separated by spaces / dots / dashes: "s t u p i d", "s.t.u.p.i.d"
SPACED_RE = re.compile(r"\b(?:[a-z][ .\-_]){3,}[a-z]\b")
# same character 3+ times -> keep 2: "stuuuupid" -> "stuupid", "soooo" -> "soo"
REPEAT_RE = re.compile(r"(.)\1{2,}")
SPACES_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"@user|[a-z0-9*']+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = text.translate(TURKISH_MAP).lower()
    text = URL_RE.sub(" url ", text)
    text = MENTION_RE.sub(" @user ", text)
    text = HASHTAG_RE.sub(r"\1", text)
    text = LEET_INSIDE_RE.sub(lambda m: "".join(LEET_MAP[c] for c in m.group()), text)
    text = LEET_START_RE.sub("s", text)
    text = SPACED_RE.sub(lambda m: re.sub(r"[ .\-_]", "", m.group()), text)
    text = REPEAT_RE.sub(r"\1\1", text)
    return SPACES_RE.sub(" ", text).strip()


def tokenize(text: str) -> list:
    """Simple word tokenizer used by the classical models and word2vec."""
    return TOKEN_RE.findall(text)
