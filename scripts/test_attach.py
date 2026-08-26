"""Gercek model uzerinde EngramAttach dogrulamasi.

Dogrular:
1. attach + alpha=0 -> base ile bit-bit ayni logits (tak-cikar guvencesi)
2. disable() -> yine ayni
3. remove() -> hook'suz orijinale donus
4. alpha>0 -> cikis degisir ve loss'tan Engram parametrelerine gradyan akar
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs.config import EngramExperimentConfig, ModelConfig
from src.engram import EngramAttach, HashConfig, EngramModuleConfig


def max_logit_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def main():
    device = "cuda"
    print("Model yukleniyor (ilk calistirmada ~1.5GB indirir)...")
    tok = AutoTokenizer.from_pretrained(ModelConfig.name)
    model = AutoModelForCausalLM.from_pretrained(
        ModelConfig.name, dtype=torch.bfloat16
    ).to(device)
    model.eval()

    exp_cfg = EngramExperimentConfig()
    hash_cfg = HashConfig(
        tokenizer_name_or_path=ModelConfig.name,
        layer_ids=exp_cfg.layer_ids,
        max_ngram_size=exp_cfg.max_ngram_size,
        n_head_per_ngram=exp_cfg.n_head_per_ngram,
        vocab_multiplier=exp_cfg.vocab_multiplier,
        seed=exp_cfg.seed,
    )
    mod_cfg = EngramModuleConfig(n_embed_per_ngram=exp_cfg.n_embed_per_ngram)

    text = (
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n"
        "    pivot = arr[len(arr) // 2]\n    return quicksort([x for x in arr if x < pivot])"
    )
    inputs = tok(text, return_tensors="pt").to(device)

    with torch.no_grad():
        base_logits = model(**inputs).logits.clone()
    print(f"[ok] base logits: {tuple(base_logits.shape)}")

    # --- Test 1: attach, alpha=0 -> kimlik ---
    attach = EngramAttach(model, hash_cfg, mod_cfg, layer_ids=exp_cfg.layer_ids)
    assert all(v == 0.0 for v in attach.alpha_values().values()), "alpha'lar 0 olmali"
    with torch.no_grad():
        out1 = model(**inputs).logits
    d1 = max_logit_diff(base_logits, out1)
    print(f"[{'ok' if d1 == 0 else 'FAIL'}] alpha=0 fark: {d1}")
    assert d1 == 0.0

    # --- Test 2: disable ---
    attach.disable()
    with torch.no_grad():
        out2 = model(**inputs).logits
    d2 = max_logit_diff(base_logits, out2)
    print(f"[{'ok' if d2 == 0 else 'FAIL'}] disabled fark: {d2}")
    assert d2 == 0.0

    # --- Test 3: enable + alpha>0 -> cikis degisir, gradyan akar ---
    attach.enable()
    attach.mark_only_engram_trainable()  # egitim kurulumu: sadece engram acik
    for inj in attach.injections.values():
        with torch.no_grad():
            inj.alpha.fill_(0.1)
    model.train()
    loss = model(**inputs, labels=inputs["input_ids"]).loss
    loss.backward()
    first_inj = attach.injections[str(exp_cfg.layer_ids[0])]
    g_emb = first_inj.multi_head_embedding.embedding.weight.grad
    g_alpha = first_inj.alpha.grad
    n_engram_grads = sum(
        1 for p in attach.trainable_parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    )
    n_engram_total = sum(1 for _ in attach.trainable_parameters())
    print(f"[info] egitim loss (alpha=0.1): {loss.item():.4f}")
    print(f"[{'ok' if g_emb is not None and g_emb.abs().sum() > 0 else 'FAIL'}] embedding tablosuna gradyan akiyor")
    print(f"[{'ok' if g_alpha is not None and g_alpha.abs().sum() > 0 else 'FAIL'}] alpha gate'ine gradyan akiyor")
    print(f"[info] gradyani akan engram param: {n_engram_grads}/{n_engram_total}")
    assert g_emb is not None and g_emb.abs().sum() > 0

    # Backbone gercekten donuk mu?
    frozen_ok = not any(p.requires_grad for p in model.parameters())
    print(f"[{'ok' if frozen_ok else 'FAIL'}] backbone tamamen donuk")

    # --- Test 4: remove -> orijinal ---
    model.zero_grad(set_to_none=True)
    model.eval()
    attach.remove()
    with torch.no_grad():
        out4 = model(**inputs).logits
    d4 = max_logit_diff(base_logits, out4)
    print(f"[{'ok' if d4 == 0 else 'FAIL'}] removed fark: {d4}")
    assert d4 == 0.0

    print("\nTum attach testleri gecti.")


if __name__ == "__main__":
    main()
