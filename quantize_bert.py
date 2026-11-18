"""Step 5 - quantize the fine-tuned BERT to INT8 (Area D).

Dynamic quantization: the weights of every Linear layer (attention Q/K/V/output and
feed-forward layers, ~85% of BERT's compute) are stored as 8-bit integers instead of
32-bit floats. Activations are quantized on the fly at run time. Runs on CPU.
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
from moderation.models import file_mb, quantize_dynamic_int8, set_quant_engine


def main():
    src = config.MODELS_DIR / "bert_fp32"
    dst = config.MODELS_DIR / "bert_int8"
    dst.mkdir(parents=True, exist_ok=True)

    print(f"Quantization engine: {set_quant_engine()}")
    model = AutoModelForSequenceClassification.from_pretrained(src).eval()
    qmodel = quantize_dynamic_int8(model)

    torch.save(qmodel.state_dict(), dst / "model_int8.pt")
    model.config.save_pretrained(dst)
    AutoTokenizer.from_pretrained(src).save_pretrained(dst)

    fp32 = sum(file_mb(f) for f in src.iterdir() if f.suffix in (".safetensors", ".bin"))
    int8 = file_mb(dst / "model_int8.pt")
    print(f"FP32 size: {fp32:.1f} MB  ->  INT8 size: {int8:.1f} MB  ({fp32 / int8:.1f}x smaller)")
    print(f"Saved to {dst}")


if __name__ == "__main__":
    main()
