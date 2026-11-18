# Study guide: how every part works (read this before presenting)

The doctor can ask you to explain any component. This is the short explanation for each one.

## The problem
A gaming platform needs to flag offensive chat messages **instantly** on a **CPU-only** server.
Input: one message. Output: `clean` or `offensive` + confidence. Offensive messages are masked.
Metric: **macro-F1**. The classes are imbalanced (about 1/3 offensive), so plain accuracy can look
good even for a bad model. Macro-F1 = average of the F1 of each class, so both classes count equally.

## Data (`prepare_data.py`)
OLID / OffensEval 2019: about 13k English tweets labeled offensive / not offensive.
Train = used for learning, validation = for choosing the best epoch, test (860 tweets) = used **only once**
at the end, and it is the **same test set for all models**.

## Area B – Classical NLP

**Regex normalization** (`moderation/normalize.py`). Players hide bad words. Each regex undoes one trick:
- lowercase + Turkish letters: `ş→s, ı→i ...`
- leetspeak: `id1ot → idiot`, `n00b → noob` (digit/symbol between letters)
- repeated letters: `stuuuupid → stuupid` (3 or more of the same letter become 2)
- spaced letters: `s t u p i d → stupid`
- URLs → `url`, mentions → `@user`, `#hashtag → hashtag`

**Edit-distance lexicon matcher** (`moderation/lexicon.py`) = the **rule baseline**.
Minimum edit distance = the smallest number of insertions, deletions and substitutions that change one
word into another. Computed with dynamic programming: `D[i][j] = min(D[i-1][j]+1, D[i][j-1]+1, D[i-1][j-1]+cost)`.
`stupd → stupid` = 1 edit, so it matches. Short words (4 letters or less) must match exactly, otherwise
"such" would match "suck". `*` counts as any letter (`id*ot`). No training needed.

**TF-IDF + logistic regression** (`train_baselines.py`) = the **classical baseline**.
TF-IDF gives each word (and word pair) a weight: high if frequent in this message but rare in all messages.
Logistic regression learns one weight per feature and outputs a probability.
`class_weight="balanced"` makes mistakes on the smaller class (offensive) cost more.

## Area A – Neural networks (`moderation/neural.py`, `train_word2vec_ffn.py`)

**Our own word2vec** (skip-gram with negative sampling), trained only on the training tweets.
Each word gets a 100-number vector. For each word, its neighbours (window of 5 words) should have a high
dot product with it, and 5 random "negative" words should have a low one.
Loss = `-log σ(v·u_context) - Σ log σ(-v·u_negative)`. After training, similar words have similar vectors
(the script prints nearest neighbours as a check).

**Feedforward network**: message vector = average of its word vectors →
`Linear(100→64) → ReLU → Dropout(0.3) → Linear(64→2)` → softmax.
Trained with **SGD** (momentum 0.9) and **backpropagation** (`loss.backward()` computes gradients
with the chain rule, `opt.step()` updates the weights). We keep the epoch with the best validation macro-F1.

## Area C – Transformer (`train_bert.py`)
`bert-base-uncased`: WordPiece tokenizer (splits rare words into known sub-word pieces; pieces that continue a word start with `##`),
12 Transformer layers with **self-attention** (every token looks at every other token to build its meaning
in context), 110M parameters, pre-trained on books + Wikipedia.
**Fine-tuning**: we add a classification head (a Linear layer on the `[CLS]` token) and train the whole model
for 2 epochs on our tweets (AdamW, learning rate 2e-5, warm-up, weighted cross-entropy).
Why better than TF-IDF? It understands context: "you killed it!" vs "kill yourself".

## Area D – Quantization (`quantize_bert.py`)
**Dynamic INT8 quantization**: the weights of every `Linear` layer are stored as 8-bit integers instead
of 32-bit floats (scale + zero-point per tensor). Activations are converted to INT8 on the fly during inference.
Result: smaller model (the embedding table stays FP32, so it is less than 4x smaller) and faster on CPU, usually with
a very small drop in F1. We check it against our **latency budget of 50 ms per message** (`config.py`).
Latency = mean over 200 single messages after 10 warm-up runs (`evaluate.py`).

## The application (`app.py`)
Gradio web page. Type a message → normalize → quantized BERT → label + confidence → if offensive, mask the
bad words (lexicon) or hide the whole message → the side panel logs latency and whether it met the 50 ms budget.
You can switch models with the dropdown, and the "compare all models" table shows the other four models.

## Likely questions
- *Why macro-F1?* Imbalanced classes; see above.
- *Why is the INT8 model not 4x smaller?* Only Linear layers are quantized; the embedding table stays FP32.
- *What if BERT does not beat TF-IDF?* That is a valid finding (see the homework rules); explain it with your failure examples.
- *Is the test set the same for all models?* Yes, `data/test.csv`, used only in `evaluate.py`.
