"""Run the whole pipeline: data -> baselines -> word2vec+FFN -> BERT -> quantization -> evaluation.

Output is shown on screen AND appended to setup_log.txt.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "setup_log.txt"
STEPS = ["prepare_data.py", "train_baselines.py", "train_word2vec_ffn.py",
         "train_bert.py", "quantize_bert.py", "evaluate.py"]
env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")

with open(LOG, "a", encoding="utf-8") as log:
    def out(line):
        print(line, end="", flush=True)
        log.write(line)
        log.flush()

    for script in STEPS:
        out(f"\n{'=' * 70}\nRunning {script}\n{'=' * 70}\n")
        p = subprocess.Popen([sys.executable, str(ROOT / script)], cwd=ROOT, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            out(line)
        if p.wait() != 0:
            out(f"\n{script} FAILED - fix the error above and run again.\n")
            sys.exit(1)
    out("\nAll done. Start the app with:  python app.py\n")
