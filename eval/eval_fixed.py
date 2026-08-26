"""Paired (sabit blok) degerlendirme: tum modeller AYNI token bloklarinda olculur.

Rastgele crop'larin val bolgesinin zor/kolay kisimlarina dengesiz dusmesi,
kosular arasinda yapay farklar yaratıyordu (C: 1.58 vs 1.75 ayni checkpoint!).
Burada val bolgesinden sabit, ust uste binmeyen N pencere alinir;
her model ayni pencerelerde olculur -> dogrudan karsilastirilabilir.
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from configs.config import EngramExperimentConfig, ModelConfig
from configs.config import resolve_dtype
from src.data_loader import PackedTokenDataset
from src.engram import EngramAttach, EngramModuleConfig, HashConfig


def read_windows(ds: PackedTokenDataset, n_windows: int, seq_len: int):
    """Val bolgesinin SONUNDAKI n_windows adet ardısik pencere (deterministik)."""
    total_val = ds.total - ds.train_end
    usable = min(n_windows * seq_len, max(total_val - seq_len - 1, 0))
    start = ds.total - usable  # val sonundan geriye dogru
    xs = []
    pos = start
    while pos + seq_len + 1 <= ds.total:
        span = torch.from_numpy(ds._read_span(pos, seq_len + 1).astype(np.int64))
        xs.append((span[:-1], span[1:]))
        pos += seq_len
    return xs


@torch.no_grad()
def eval_windows(model, windows, device, bs=8):
    losses = []
    for i in range(0, len(windows), bs):
        chunk = windows[i : i + bs]
        x = torch.stack([w[0] for w in chunk]).to(device)
        y = torch.stack([w[1] for w in chunk]).to(device)
        # token-basina loss (uzunluklar esit oldugu icin ortalama dogru)
        l = model(input_ids=x, labels=x).loss.item()
        losses.append(l)
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/python_1b")
    ap.add_argument("--val-tokens", type=int, default=10_000_000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--n-windows", type=int, default=256,
                    help="val sonundaki sabit pencere sayisi (256x512 = 131K token)")
    ap.add_argument("--lora", default=None, help="LoRA adapter klasoru (atlamak icin: none)")
    ap.add_argument("--engrams", nargs="*", default=[],
                    help="label=checkpoint.pt ciftleri, orn: C2=runs/C2_python/engram_step6000.pt")
    ap.add_argument("--out", default=None, help="sonuclari bu dosyaya da yaz")
    args = ap.parse_args()

    device = "cuda"
    ds = PackedTokenDataset(args.data_dir, val_tokens=args.val_tokens)
    windows = read_windows(ds, args.n_windows, args.seq_len)
    lines = [f"Sabit eval seti: {len(windows)} pencere x {args.seq_len} tok = "
             f"{len(windows)*args.seq_len:,} token", ""]

    results = {}

    # --- A ---
    model = AutoModelForCausalLM.from_pretrained(ModelConfig.name, dtype=resolve_dtype()).to(device).eval()
    results["A - Base"] = eval_windows(model, windows, device)
    del model
    torch.cuda.empty_cache()

    # --- B ---
    if args.lora and args.lora.lower() != "none":
        from peft import PeftModel
        model = AutoModelForCausalLM.from_pretrained(ModelConfig.name, dtype=resolve_dtype()).to(device)
        model = PeftModel.from_pretrained(model, args.lora).eval()
        results["B - LoRA"] = eval_windows(model, windows, device)
        del model
        torch.cuda.empty_cache()

    # --- Engram kosulari ---
    from src.engram import EngramAttach, EngramModuleConfig, HashConfig
    exp = EngramExperimentConfig()
    for spec in args.engrams:
        label, ckpt_path = spec.split("=", 1)
        model = AutoModelForCausalLM.from_pretrained(ModelConfig.name, dtype=resolve_dtype()).to(device)
        h = HashConfig(
            tokenizer_name_or_path=ModelConfig.name,
            layer_ids=exp.layer_ids,
            max_ngram_size=exp.max_ngram_size,
            n_head_per_ngram=exp.n_head_per_ngram,
            vocab_multiplier=exp.vocab_multiplier,
            seed=exp.seed,
        )
        m = EngramModuleConfig(
            n_embed_per_ngram=exp.n_embed_per_ngram,
            n_head_per_ngram=exp.n_head_per_ngram,
        )
        attach = EngramAttach(model, h, m, layer_ids=exp.layer_ids)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        attach.injections.load_state_dict(ckpt["injections"])
        attach.enable()
        model.eval()
        results[label] = eval_windows(model, windows, device)
        del model, attach
        torch.cuda.empty_cache()

    base = results["A - Base"]
    lines.append(f"{'kosu':<14} {'loss':>8} {'ppl':>8} {'delta':>9}")
    lines.append("-" * 44)
    for k, v in results.items():
        lines.append(f"{k:<14} {v:>8.4f} {math.exp(v):>8.2f} {(v-base):>+9.4f}")

    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()
