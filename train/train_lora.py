"""B kosi: LoRA baseline — ayni veri, esit parametre butcesi (~113M).

Butce kalibrasyonu: r otomatik secilir; hedef C kosusundaki egitilebilir param sayisidir.
Kullanim:
    python train/train_lora.py --data-dir data/python_1b --steps 2000 --target-params 113254659
"""

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from configs.config import ModelConfig
from configs.config import resolve_dtype
from src.data_loader import PackedTokenDataset


def cosine_lr(step: int, total: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * t))


def pick_rank(model, target_params: int) -> int:
    """r'yi kabaca hedef parametre sayisina kalibre et (q,k,v,o,gate,up,down)."""
    # Birim r basina param: katman basina sum(in+out) x katman sayisi
    per_layer = 0
    for name, mod in model.model.layers[0].named_modules():
        if isinstance(mod, torch.nn.Linear):
            per_layer += mod.in_features + mod.out_features
    total_layers = len(model.model.layers)
    unit = per_layer * total_layers
    r = max(1, round(target_params / unit))
    print(f"Birim r basina ~{unit:,} param -> r = {r}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/python_1b")
    ap.add_argument("--val-tokens", type=int, default=10_000_000)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-interval", type=int, default=20)
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--out-dir", default="runs/B_lora")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-params", type=int, default=None,
                    help="C ile esitlemek icin; verilmezse r=64")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Model yukleniyor...")
    model = AutoModelForCausalLM.from_pretrained(ModelConfig.name, dtype=resolve_dtype()).to(device)
    model.config.use_cache = False

    if args.target_params:
        r = pick_rank(model, args.target_params)
    else:
        r = 64

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=r * 2,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    # DIKKAT: GC peft sarmalamasindan SONRA etkinlestirilmali; once cagirilirsa
    # sarmalama GC'yi etkisiz birakiyor ve aktivasyonlar patliyordu (RAM spill bug'i).
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()  # GC yalnizca training modunda aktif; from_pretrained eval() ile yukluyor
    model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    print("Veri aciliyor...")
    ds = PackedTokenDataset(args.data_dir, val_tokens=args.val_tokens)
    print(f"Toplam {ds.total:,} token (train: {ds.train_end:,} | val: {ds.total - ds.train_end:,})")

    warmup = int(0.03 * args.steps)

    @torch.no_grad()
    def evaluate(n_batches: int = 8) -> float:
        model.eval()
        losses = []
        for _ in range(n_batches):
            x, _ = ds.get_batch(args.bsz, args.seq_len, rng, split="val")
            x = x.to(device)
            losses.append(model(input_ids=x, labels=x).loss.item())
        model.train()
        return sum(losses) / len(losses)

    t0 = time.time()
    tokens_seen = 0
    log = []

    for step in range(1, args.steps + 1):
        lr = cosine_lr(step, args.steps, args.lr, warmup)
        for g in opt.param_groups:
            g["lr"] = lr

        x, _ = ds.get_batch(args.bsz, args.seq_len, rng)
        x = x.to(device)
        loss = model(input_ids=x, labels=x).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

        tokens_seen += x.numel()
        if step % args.log_interval == 0 or step == 1:
            dt = time.time() - t0
            msg = (f"step {step:5d}/{args.steps} | loss {loss.item():.4f} | lr {lr:.2e} | "
                   f"{tokens_seen/dt:,.0f} tok/s")
            print(msg)
            log.append(msg)

        if step % args.eval_interval == 0 or step == args.steps:
            vl = evaluate()
            msg = f"step {step:5d} | VAL LOSS {vl:.4f} (ppl {math.exp(vl):.2f})"
            print(msg)
            log.append(msg)
            model.save_pretrained(out_dir / f"adapter_step{step}")

    with open(out_dir / "log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log))
    print(f"\nBitti. Adapter'lar: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
