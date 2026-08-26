"""Kaggle script-kernel: Engram C3 tam zincir (prep -> train -> eval).

GPU: P100/T4 x2 otomatik. bf16 desteklenmiyorsa fp16'ya duser (resolve_dtype).
HF token: Kaggle Secret "ox" (yoksa HF_TOKEN env degiskeni).
"""

import os
import subprocess
import sys

REPO = "https://github.com/EminGNC/engram-domain-memory.git"
WD = "/kaggle/working"

# --- HF token: once secret, sonra yerel dosya (git'e girmez), sonra env ---
try:
    from kaggle_secrets import UserSecretsClient

    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("ox")
    print("HF_TOKEN: kaggle secret 'ox' alindi")
except Exception:
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from token_local import HF_TOKEN  # gitignored yerel dosya

        os.environ["HF_TOKEN"] = HF_TOKEN
        print("HF_TOKEN: token_local.py alindi")
    except Exception as e:
        if os.environ.get("HF_TOKEN"):
            print("HF_TOKEN: env degiskeninden alindi")
        else:
            print(f"UYARI: token hicbir kaynaktan gelmedi ({e})")

# --- Repo + bagimliliklar ---
os.chdir(WD)
if not os.path.exists("engram-domain-memory"):
    subprocess.run(["git", "clone", REPO], check=True)
REPO_DIR = f"{WD}/engram-domain-memory"
sys.path.insert(0, REPO_DIR)  # chdir sys.path'i degistirmez -> importlar icin sart
os.chdir(REPO_DIR)

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "-r", "cloud/requirements.txt"], check=True)

import torch  # noqa: E402

print("cuda:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0)
      if torch.cuda.is_available() else "-")
print("gpu bf16 destekliyor mu:", torch.cuda.is_bf16_supported()
      if torch.cuda.is_available() else "-")

if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    if cap[0] < 7:
        raise SystemExit(f"BU GPU (sm_{cap[0]}{cap[1]}) PyTorch tarafindan desteklenmiyor! "
                         "Accelerator'u T4 x2 yapip yeniden dene.")
from configs.config import resolve_dtype  # noqa: E402

print("calistirma dtype:", resolve_dtype())


def run(phase):
    print(f"\n===== FAZ: {phase} =====", flush=True)
    r = subprocess.run([sys.executable, "cloud/run_c3.py", phase])
    if r.returncode != 0:
        raise SystemExit(f"FAZ BASARISIZ: {phase}")


run("prep")     # ~1-2 saat
run("train")    # ~2-3 saat
run("eval")     # ~15 dk

print("\n=== C3 ZINCIRI TAMAMLANDI ===")
