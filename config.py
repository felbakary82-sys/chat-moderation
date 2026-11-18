"""Shared settings for every script. Change values here, not inside the scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# Dataset: OLID (OffensEval 2019, English tweets) as packaged by TweetEval on Hugging Face
HF_DATASET = "cardiffnlp/tweet_eval"
HF_CONFIG = "offensive"
LABELS = ["clean", "offensive"]          # 0 = clean, 1 = offensive

# Transformer
BERT_NAME = "bert-base-uncased"
MAX_LEN = 64                              # tweets are short; 64 sub-word tokens is enough
BERT_EPOCHS = 2
BERT_BATCH = 32
BERT_LR = 2e-5

# Our own word2vec + feedforward network
W2V_DIM = 100
W2V_WINDOW = 5
W2V_NEGATIVES = 5
W2V_EPOCHS = 5
W2V_MIN_COUNT = 2
FFN_HIDDEN = 64
FFN_EPOCHS = 30

# Latency budget we set for the CPU server (Area D requirement)
LATENCY_BUDGET_MS = 50
LATENCY_RUNS = 200                        # requirement: average over >= 100 runs

SEED = 42

# Google Drive file id of models.zip (fill in after you upload it, see README)
DRIVE_FILE_ID = ""
