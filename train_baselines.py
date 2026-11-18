"""Step 2 - classical baselines (Area B).

  1. Rule baseline: regex normalization + edit-distance lexicon matcher (no training)
  2. TF-IDF (words + bigrams) + logistic regression
"""
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

import config
from moderation.models import RuleModel
from moderation.normalize import normalize, tokenize


def main():
    train = pd.read_csv(config.DATA_DIR / "train.csv")
    val = pd.read_csv(config.DATA_DIR / "validation.csv")
    x_train = [normalize(t) for t in train["text"]]
    x_val = [normalize(t) for t in val["text"]]

    # 1. rule baseline - nothing to train, just check it on validation
    rule = RuleModel()
    rule_pred = [rule.predict(t)[0] for t in val["text"]]
    print(f"Rule baseline   validation macro-F1: {f1_score(val['label'], rule_pred, average='macro'):.4f}")

    # 2. tf-idf + logistic regression
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                                  ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0)),
    ])
    pipe.fit(x_train, train["label"])
    pred = pipe.predict(x_val)
    print(f"TF-IDF + LogReg validation macro-F1: {f1_score(val['label'], pred, average='macro'):.4f}")

    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, config.MODELS_DIR / "tfidf_lr.joblib")
    print("Saved models/tfidf_lr.joblib")


if __name__ == "__main__":
    main()
