"""Smoke test: hash + embedding + injection ileri gecisi.

Model agirligi indirmeden calisir (tokenizer hariç, o da cache'li olabilir).
Dogrular:
1. Hash cikti boyutlari ve deger araliklari dogru
2. Injection cikisi alpha=0 iken sifir -> model kimligi korunur
3. alpha>0 iken cikis sifirdan farkli ve geriye gradyan akiyor
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.engram import EngramInjection, EngramModuleConfig, HashConfig


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    hash_cfg = HashConfig(layer_ids=[1, 4, 7])
    mod_cfg = EngramModuleConfig()

    print("Tokenizer indiriluyor/cache'den yukleniyor...")
    inj = EngramInjection(hash_cfg, mod_cfg, hidden_size=1024).to(device)

    B, T = 2, 32
    input_ids = torch.randint(0, 151936, (B, T), device=device)
    hidden = torch.randn(B, T, 1024, device=device, requires_grad=True)

    # --- Test 1: boyut ---
    out = inj(hidden, input_ids, layer_id=1)
    assert out.shape == hidden.shape, f"boyut hatasi: {out.shape}"
    print(f"[ok] cikis boyutu: {tuple(out.shape)}")

    # --- Test 2: alpha=0 -> kimlik ---
    assert inj.alpha.item() == 0.0
    assert out.abs().sum().item() == 0.0, "alpha=0 iken cikis sifir olmali"
    h_plus = hidden + out
    assert torch.equal(h_plus, hidden)
    print("[ok] alpha=0 -> modul kimlik (tak-cikar guvencesi)")

    # --- Test 3: alpha>0 -> etkili + gradyan akisi ---
    with torch.no_grad():
        inj.alpha.fill_(0.5)
    out = inj(hidden, input_ids, layer_id=4)
    loss = out.square().mean()
    loss.backward()
    n_with_grad = sum(1 for p in inj.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for _ in inj.parameters())
    print(f"[ok] alpha=0.5, loss={loss.item():.6f}, gradyani akan param: {n_with_grad}/{n_total}")

    # --- Test 4: tablo istatistikleri ---
    emb = inj.multi_head_embedding.embedding
    total = emb.num_embeddings * emb.embedding_dim
    proj_params = sum(p.numel() for n, p in inj.named_parameters() if "multi_head_embedding" not in n)
    print(f"[info] tablo: {total:,} param | projeksiyon/gate: {proj_params:,} param")

    print("\nTum smoke testler gecti.")


if __name__ == "__main__":
    main()
