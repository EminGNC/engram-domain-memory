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
    # Token butcesi sabit (~16.5M); OOM olursa otomatik kucuk konfigurasyona duser.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    budget = 16_500_000
    configs = [(8, 512), (6, 512), (12, 256), (4, 512)]
    for bsz, seq in configs:
        steps = max(1, round(budget / (bsz * seq)))
        eval_iv = max(200, round(steps / 6))
        print(f"\n===== FAZ: train (bsz{bsz} x seq{seq} x {steps} adim) =====", flush=True)
        r = subprocess.run([
            sys.executable, "train/train_engram.py",
            "--data-dir", DATA_PY, "--steps", str(steps), "--bsz", str(bsz),
            "--seq-len", str(seq), "--eval-interval", str(eval_iv),
            "--alpha-init", "0.05", "--table-lr", "5e-4",
            "--out-dir", f"{ROOT}/runs/C3_python",
        ])
        if r.returncode == 0:
            print("train tamamlandi:", f"bsz{bsz} x seq{seq}")
            return
        print(f"konfigurasyon basarisiz (bsz{bsz}x{seq}), bir sonrakine geciliyor...")
    raise SystemExit("tum konfigurasyonlar OOM oldu!")


def eval_all():
    import glob as _glob

    cks = sorted(_glob.glob(f"{ROOT}/runs/C3_python/engram_step*.pt"))
    if not cks:
        raise SystemExit("C3 checkpoint bulunamadi!")
    ck = cks[-1]
    print("eval edilen checkpoint:", ck)
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
