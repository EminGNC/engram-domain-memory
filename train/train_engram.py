"""C kosi: donuk backbone + sadece Engram parametreleri egitimi.

Kullanim:
    python train/train_engram.py --data-dir data/python_1b --steps 2000
"""

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from configs.config import EngramExperimentConfig, ModelConfig
from configs.config import resolve_dtype
from src.data_loader import PackedTokenDataset
from src.engram import EngramAttach, HashConfig, EngramModuleConfig


def cosine_lr(step: int, total: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/python_1b")
    ap.add_argument("--val-tokens", type=int, default=10_000_000)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--table-lr", type=float, default=5e-4,
                    help="embedding tablosu icin ayrik LR (olu-gate kirisini kirmak icin)")
    ap.add_argument("--alpha-init", type=float, default=0.05,
                    help="dis gate baslangic degeri (0 = eski davranis)")
    ap.add_argument("--drift-check-step", type=int, default=500,
                    help="tablo norm drift erken-uyari kontrolunun yapilacagi adim")
    ap.add_argument("--freeze-table", action="store_true",
                    help="E2 kosulu: tabloyu rastgele init ile dondur, sadece reader/gate egit")
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-interval", type=int, default=20)
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--out-dir", default="runs/engram_c")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_cfg = EngramExperimentConfig(lr=args.lr, batch_size=args.bsz, seq_len=args.seq_len)

    print("Model yukleniyor...")
    model = AutoModelForCausalLM.from_pretrained(ModelConfig.name, dtype=resolve_dtype()).to(device)
    model.gradient_checkpointing_enable()  # 8GB kart icin guvenli

    hash_cfg = HashConfig(
        tokenizer_name_or_path=ModelConfig.name,
        layer_ids=exp_cfg.layer_ids,
        max_ngram_size=exp_cfg.max_ngram_size,
        n_head_per_ngram=exp_cfg.n_head_per_ngram,
        vocab_multiplier=exp_cfg.vocab_multiplier,
        seed=exp_cfg.seed,
    )
    mod_cfg = EngramModuleConfig(
        n_embed_per_ngram=exp_cfg.n_embed_per_ngram,
        n_head_per_ngram=exp_cfg.n_head_per_ngram,
        alpha_init=args.alpha_init,
    )

    attach = EngramAttach(model, hash_cfg, mod_cfg, layer_ids=exp_cfg.layer_ids)
    attach.enable()
    attach.mark_only_engram_trainable()
    if args.freeze_table:
        # E2: rastgele icerik kontrolu ??? tablo donuk, mekanizma ogreniyor
        for inj in attach.injections.values():
            inj.multi_head_embedding.embedding.weight.requires_grad_(False)
        print(">>> E2 MODU: tablo rastgele ve DONUK")
    model.train()

    # --- SIGORTA 1 (Bug #8): egitilebilir her param fp32 olmali ---
    bad_dtype = [n for inj in attach.injections.values()
                 for n, p in inj.named_parameters() if p.dtype != torch.float32]
    if bad_dtype:
        raise SystemExit(f"FP32 IHLALI (Bug #8 geri dondu!): {bad_dtype} -> durduruluyor")
    print("[sigorta] tum egitilebilir parametreler fp32 [OK]")

    # --- SIGORTA 2 (Bug #8/9): tablo drift erken-uyari ---
    init_table_norm = float(attach.injections[str(exp_cfg.layer_ids[0])]
                            .multi_head_embedding.embedding.weight.float().norm(dim=1).mean())

    # Ayrik LR gruplari: tablo hizli ogrensin, projeksiyon/gateler daha temkinli
    table_params, other_params = [], []
    for inj in attach.injections.values():
        for name, p in inj.named_parameters():
            (table_params if "multi_head_embedding" in name else other_params).append(p)
    params = table_params + other_params
    opt = torch.optim.AdamW(
        [
            {"params": table_params, "lr": args.table_lr},
            {"params": other_params, "lr": args.lr},
        ],
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    base_lrs = [g["lr"] for g in opt.param_groups]
    n_trainable = sum(p.numel() for g in opt.param_groups for p in g["params"])
    print(f"Egitilebilir engram param: {n_trainable:,} "
          f"(tablo: {sum(p.numel() for p in table_params):,}, diger: {sum(p.numel() for p in other_params):,})")

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
        lr_factor = cosine_lr(step, args.steps, 1.0, warmup)  # oran katsayisi
        for g, base in zip(opt.param_groups, base_lrs):
            g["lr"] = base * lr_factor

        x, y = ds.get_batch(args.bsz, args.seq_len, rng)
        x = x.to(device)
        # DIKKAT: transformers labels'i ic olarak kaydirir -> labels=x dogru kullanim.
        # (y'yi verirsek cift shift olur; bu hata smoke testte yakalandi.)
        loss = model(input_ids=x, labels=x).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

        tokens_seen += x.numel()
        if step == args.drift_check_step and not args.freeze_table:
            cur = float(attach.injections[str(exp_cfg.layer_ids[0])]
                        .multi_head_embedding.embedding.weight.float().norm(dim=1).mean())
            drift = abs(cur - init_table_norm)
            if drift < 0.005:
                print(f"\n!!! ERKEN DRIFT UYARISI: tablo normu {init_table_norm:.4f} -> "
                      f"{cur:.4f} (drift {drift:.5f}) ??? ogrenme yine kisik olabilir! "
                      f"(Bug #8 kontrolu: parametreler fp32 mi?)\n")
            else:
                print(f"[sigorta] step {step}: tablo drift {drift:.4f} [OK] ogrenme canli")
        if step % args.log_interval == 0 or step == 1:
            dt = time.time() - t0
            tps = tokens_seen / dt
            msg = (f"step {step:5d}/{args.steps} | loss {loss.item():.4f} | "
                   f"lr*{lr_factor:.3f} | {tps:,.0f} tok/s")
            print(msg)
            log.append(msg)

        if step % args.eval_interval == 0 or step == args.steps:
            vl = evaluate()
            msg = f"step {step:5d} | VAL LOSS {vl:.4f} (ppl {math.exp(vl):.2f})"
            print(msg)
            log.append(msg)
            ckpt = {
                "step": step,
                "injections": attach.injections.state_dict(),
                "config": {
                    "layer_ids": exp_cfg.layer_ids,
                    "max_ngram_size": exp_cfg.max_ngram_size,
                    "n_head_per_ngram": exp_cfg.n_head_per_ngram,
                    "vocab_multiplier": exp_cfg.vocab_multiplier,
                    "n_embed_per_ngram": exp_cfg.n_embed_per_ngram,
                    "seed": exp_cfg.seed,
                },
            }
            torch.save(ckpt, out_dir / f"engram_step{step}.pt")
            sz = (out_dir / f"engram_step{step}.pt").stat().st_size / 2**20
            print(f"[sigorta] checkpoint yazildi: {sz:.0f} MB "
                  f"({'makul [OK]' if 50 < sz < 2000 else 'SISE SOR!!!'})")

    with open(out_dir / "log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log))

    # Gate degerlerini raporla (analiz icin)
    print("\nGate (alpha) degerleri:")
    for lid, v in attach.alpha_values().items():
        print(f"  katman {lid}: alpha={v:.6f}")

    print(f"\nBitti. Checkpoint'ler: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
