# Results on the held-out test set (860 messages)

Latency: mean over 200 single-message runs (after 10 warm-up runs).

Hardware: Windows 11, CPU: Intel64 Family 6 Model 140 Stepping 1, GenuineIntel, 8 logical cores, torch 2.8.0+cpu (4 threads), CPU only

| Model | Precision | Macro-F1 | Accuracy | Size (MB) | Latency mean (ms) | Latency p95 (ms) | Within 50 ms budget |
|---|---|---|---|---|---|---|---|
| Rule (regex + edit distance) | rules | 0.7006 | 0.7779 | 0.001 | 0.04 | 0.08 | yes |
| TF-IDF + LogReg | FP64 | 0.7291 | 0.7779 | 0.985 | 0.52 | 0.6 | yes |
| Word2Vec + FFN | FP32 | 0.6759 | 0.7151 | 3.595 | 0.2 | 0.27 | yes |
| BERT FP32 | FP32 | 0.8052 | 0.8407 | 437.959 | 79.47 | 106.99 | no |
| BERT INT8 (quantized) | INT8 weights | 0.7761 | 0.8419 | 181.495 | 23.39 | 33.46 | yes |
