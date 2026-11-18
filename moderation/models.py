"""One common interface for all five models: predict(text) -> (label, confidence).

Both evaluate.py and app.py use these classes, so the numbers in the results
table come from exactly the same code that runs in the application.
Every model receives the same normalized text (see normalize.py).
"""
import os
import platform
from pathlib import Path

from .lexicon import LEXICON_PATH, LexiconMatcher
from .normalize import normalize, tokenize


def file_mb(path) -> float:
    path = Path(path)
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
    return path.stat().st_size / 1e6


class RuleModel:
    name = "Rule (regex + edit distance)"
    precision = "rules"

    def __init__(self):
        self.matcher = LexiconMatcher()
        self.size_mb = file_mb(LEXICON_PATH)

    def predict(self, text):
        hits = self.matcher.find(normalize(text))
        if not hits:
            return 0, 0.60                       # no bad word found -> clean
        best = min(d for _, _, d in hits)
        return 1, (0.95 if best == 0 else 0.75)  # exact match is more certain than a typo match


class TfidfModel:
    name = "TF-IDF + LogReg"
    precision = "FP64"

    def __init__(self, models_dir):
        import joblib
        path = Path(models_dir) / "tfidf_lr.joblib"
        self.pipe = joblib.load(path)
        self.size_mb = file_mb(path)

    def predict(self, text):
        proba = self.pipe.predict_proba([normalize(text)])[0]
        label = int(proba.argmax())
        return label, float(proba[label])


class W2VFFNModel:
    name = "Word2Vec + FFN"
    precision = "FP32"

    def __init__(self, models_dir):
        import torch
        from .neural import FeedForward
        path = Path(models_dir) / "w2v_ffn.pt"
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.torch = torch
        self.stoi = {w: i for i, w in enumerate(ckpt["vocab"])}
        self.emb = ckpt["emb"]
        self.ffn = FeedForward(ckpt["dim"], ckpt["hidden"])
        self.ffn.load_state_dict(ckpt["ffn"])
        self.ffn.eval()
        self.size_mb = file_mb(path)

    def predict(self, text):
        from .neural import average_vectors
        x = average_vectors([tokenize(normalize(text))], self.stoi, self.emb)
        with self.torch.inference_mode():
            p = self.torch.softmax(self.ffn(x), -1)[0]
        label = int(p.argmax())
        return label, float(p[label])


def set_quant_engine():
    import torch
    engines = torch.backends.quantized.supported_engines
    for e in ("x86", "fbgemm", "qnnpack"):
        if e in engines:
            torch.backends.quantized.engine = e
            return e
    return torch.backends.quantized.engine


def quantize_dynamic_int8(model):
    """Replace every nn.Linear by an INT8 dynamically-quantized Linear."""
    import torch
    set_quant_engine()
    try:
        from torch.ao.quantization import quantize_dynamic
    except ImportError:
        from torch.quantization import quantize_dynamic
    return quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


class BertModel:
    """Fine-tuned BERT. int8=True loads the dynamically quantized version."""

    def __init__(self, models_dir, int8=False, max_len=64):
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.max_len = max_len
        if int8:
            path = Path(models_dir) / "bert_int8"
            self.name, self.precision = "BERT INT8 (quantized)", "INT8 weights"
            config = AutoConfig.from_pretrained(path)
            model = AutoModelForSequenceClassification.from_config(config)
            model.eval()
            model = quantize_dynamic_int8(model)
            state = torch.load(path / "model_int8.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(state)
            self.size_mb = file_mb(path / "model_int8.pt")
        else:
            path = Path(models_dir) / "bert_fp32"
            self.name, self.precision = "BERT FP32", "FP32"
            model = AutoModelForSequenceClassification.from_pretrained(path)
            weights = [f for f in path.iterdir() if f.suffix in (".safetensors", ".bin")]
            self.size_mb = sum(file_mb(f) for f in weights)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = model.eval()

    def predict(self, text):
        enc = self.tokenizer(normalize(text), truncation=True, max_length=self.max_len,
                             return_tensors="pt")
        with self.torch.inference_mode():
            p = self.torch.softmax(self.model(**enc).logits, -1)[0]
        label = int(p.argmax())
        return label, float(p[label])


def load_models(models_dir, verbose=True):
    """Load every model that exists. Missing ones are skipped with a message."""
    models = {}
    loaders = [
        ("rule", lambda: RuleModel()),
        ("tfidf", lambda: TfidfModel(models_dir)),
        ("w2v_ffn", lambda: W2VFFNModel(models_dir)),
        ("bert_fp32", lambda: BertModel(models_dir, int8=False)),
        ("bert_int8", lambda: BertModel(models_dir, int8=True)),
    ]
    for key, make in loaders:
        try:
            m = make()
            models[m.name] = m
            if verbose:
                print(f"  loaded  {m.name:32s} {m.size_mb:8.1f} MB")
        except Exception as e:  # noqa: BLE001 - we want to keep going
            if verbose:
                print(f"  SKIPPED {key:32s} ({type(e).__name__}: {e})")
    return models


def hardware_info():
    try:
        import torch
        threads = torch.get_num_threads()
        tv = torch.__version__
    except ImportError:
        threads, tv = "n/a", "n/a"
    cpu = platform.processor() or platform.machine()
    return (f"{platform.system()} {platform.release()}, CPU: {cpu}, "
            f"{os.cpu_count()} logical cores, torch {tv} ({threads} threads), CPU only")
