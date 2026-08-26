"""Kaggle/Colab icin tek-dosya calistirici.

Fazlar halinde calisir; oturum koparsa ayni faz tekrar tetiklenebilir
(data ve checkpoint'ler /kaggle/working altinda kalici):

    python cloud/run_c3.py prep      # veri paketleme (python + general, ~1-2 saat)
    python cloud/run_c3.py train     # C3 egitimi (~2-3 saat T4'te)
    python cloud/run_c3.py eval      # paired eval + capraz-domain

HF gated dataset (starcoderdata) icin ortam degiskeni: HF_TOKEN
"""

import os
import subprocess
import sys

PHASE = sys.argv[1] if len(sys.argv) > 1 else "help"
ROOT = "/kaggle/working" if os.path.exists("/kaggle") else "."
DATA_PY = f"{ROOT}/data/python_1b"
DATA_GEN = f"{ROOT}/data/general_1b"


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"FAILE: {cmd}")


def prep():
    os.makedirs(f"{ROOT}/src", exist_ok=True)
    # kod zaten repo'dan kopyalanmis olmali (src/, scripts/, configs/)
    if not os.path.exists(DATA_PY):
        run([sys.executable, "scripts/data/prepare_python.py",
             "--target-tokens", "1000000000", "--out-dir", DATA_PY])
    if not os.path.exists(DATA_GEN):
        run([sys.executable, "scripts/data/prepare_python.py",
             "--dataset", "HuggingFaceFW/fineweb", "--config", "sample/10BT",
             "--content-key", "text",
             "--target-tokens", "1000000000", "--out-dir", DATA_GEN])


def train():
    run([sys.executable, "train/train_engram.py",
         "--data-dir", DATA_PY, "--steps", "8000", "--bsz", "4", "--seq-len", "512",
         "--eval-interval", "2000", "--alpha-init", "0.05", "--table-lr", "5e-4",
         "--out-dir", f"{ROOT}/runs/C3_python"])


def eval_all():
    ck = f"{ROOT}/runs/C3_python/engram_step8000.pt"
    run([sys.executable, "eval/eval_fixed.py",
         "--data-dir", DATA_PY,
         "--lora", "none",
         "--engrams", f"C3={ck}",
         "--out", f"{ROOT}/runs/overnight/eval_C3_python_block.txt"])
    run([sys.executable, "eval/eval_fixed.py",
         "--data-dir", DATA_GEN,
         "--engrams", f"C3py={ck}",
         "--out", f"{ROOT}/runs/overnight/eval_C3_general_block.txt"])
    # referans noktalari icin base'i de olc:
    run([sys.executable, "eval/eval_fixed.py",
         "--lora", "none",
         "--out", f"{ROOT}/runs/overnight/eval_base_cloud.txt"])


if PHASE == "prep":
    prep()
elif PHASE == "train":
    train()
elif PHASE == "eval":
    eval_all()
else:
    print("kullanim: python cloud/run_c3.py [prep|train|eval]")
