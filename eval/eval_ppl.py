"""Perplexity degerlendirme: bir checkpoint'in (veya base'in) belirli veri uzerinde ppl'i.

Kullanim:
    # base referans
    python eval/eval_ppl.py --data-dir data/python_1b

    # engram checkpoint'i takili
    python eval/eval_ppl.py --data-dir data/python_1b --ckpt runs/engram_c/engram_step2000.pt
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs.config import ModelConfig
from src.data_loader import PackedTokenDataset
from src.engram import EngramAttach, EngramModuleConfig, HashConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/python_1b")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--val-tokens", type=int, default=10_000_000)
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--n-batches", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    device = "cuda"
    rng = np.random.default_rng(args.seed)

    model = AutoModelForCausalLM.from_pretrained(ModelConfig.name, dtype=torch.bfloat16).to(device)

    attach = None
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        hash_cfg = HashConfig(
            tokenizer_name_or_path=ModelConfig.name,
            layer_ids=cfg["layer_ids"],
            max_ngram_size=cfg["max_ngram_size"],
            n_head_per_ngram=cfg["n_head_per_ngram"],
            vocab_multiplier=cfg["vocab_multiplier"],
            seed=cfg["seed"],
        )
        mod_cfg = EngramModuleConfig(n_embed_per_ngram=cfg["n_embed_per_ngram"])
        attach = EngramAttach(model, hash_cfg, mod_cfg, layer_ids=cfg["layer_ids"])
        attach.injections.load_state_dict(ckpt["injections"])
        attach.enable()
        print(f"Checkpoint yuklendi: {args.ckpt} (step {ckpt['step']})")
        print(f"Gate degerleri: {attach.alpha_values()}")
    else:
        print("Checkpoint yok -> BASE referans olcumu")

    model.eval()
    ds = PackedTokenDataset(args.data_dir, val_tokens=args.val_tokens)

    losses = []
    with torch.no_grad():
        for i in range(args.n_batches):
            x, _ = ds.get_batch(args.bsz, args.seq_len, rng, split="val")
            x = x.to(device)
            losses.append(model(input_ids=x, labels=x).loss.item())
            if (i + 1) % 8 == 0:
                print(f"  batch {i+1}/{args.n_batches}: running loss {sum(losses)/len(losses):.4f}")

    loss = sum(losses) / len(losses)
    print(f"\nVAL LOSS: {loss:.4f} | PPL: {math.exp(loss):.4f}")


if __name__ == "__main__":
    main()
