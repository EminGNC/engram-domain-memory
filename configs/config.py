"""Deney konfigurasyonu. Tum kosular bu degerlerden turer.

Not: Sayilar yer tutucudur; pilot sonrasi kalibre edilir (bkz. plan dokumani Bolum 9).
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen3-0.6B"
    dtype: str = "bfloat16"
    device: str = "cuda"


def resolve_dtype():
    """GPU bf16 desteklemiyorsa (T4/P100/Turing ve öncesi) fp16'ya düş.

    Bulut GPU'larda (Kaggle T4/P100) bfloat16 matmul yoktur; bu yüzden
    model yüklemede sabit 'bfloat16' yerine bu fonksiyon kullanılmalı.
    """
    import torch

    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


@dataclass
class EngramExperimentConfig:
    # Hash / tablo
    layer_ids: List[int] = field(default_factory=lambda: [1, 4, 7])  # erken katmanlar
    max_ngram_size: int = 3
    n_head_per_ngram: int = 2          # v2: 8 -> 4 kolon (collision azaltma)
    vocab_multiplier: int = 24         # v2: 2 -> 24 (adres uzayi 12x buyuk)
    n_embed_per_ngram: int = 24        # head_dim = 24/2 = 12 (iso-param ~123M)
    seed: int = 0

    # Egitim
    lr: float = 1e-4
    table_lr: float = 5e-4             # v2: tabloya ayrik yuksek LR (olu-gate kirisini kirma)
    batch_size: int = 8
    seq_len: int = 1024
    grad_accum: int = 4
    warmup_ratio: float = 0.03


# Tablo buyutu hesabi (butce kontrolu icin yardimci)
def estimate_table_params(vocab_size: int, cfg: EngramExperimentConfig) -> dict:
    """Head basina asal moduller ~ vocab*multiplier oldugundan yaklasik hesap."""
    base = vocab_size * cfg.vocab_multiplier
    heads_per_layer = (cfg.max_ngram_size - 1) * cfg.n_head_per_ngram
    head_dim = cfg.n_embed_per_ngram // cfg.n_head_per_ngram
    table = heads_per_layer * base * head_dim  # katmanlar tabloyu PAYLASIR (tek Engram cekirdegi)
    proj = (
        ((cfg.max_ngram_size - 1) * cfg.n_embed_per_ngram + 1) * 1024 * 2 * len(cfg.layer_ids)  # value+key proj
        + heads_per_layer * base * head_dim  # embedding paylasimli, tek sayilir
    )
    return {
        "embedding_table": table,
        "approx_total_trainable": table + proj,
        "heads_per_layer": heads_per_layer,
        "head_dim": head_dim,
    }


if __name__ == "__main__":
    from transformers import AutoConfig

    hf_cfg = AutoConfig.from_pretrained(ModelConfig.name)
    est = estimate_table_params(hf_cfg.vocab_size, EngramExperimentConfig())
    for k, v in est.items():
        print(f"{k}: {v:,}" if isinstance(v, int) else f"{k}: {v}")
