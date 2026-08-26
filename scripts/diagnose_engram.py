"""C checkpoint'i icin teshis: tablo doluluk orani, kullanim yogunlasmasi, sinyal gucu.

Sorular:
1. Egitim verisindeki n-gram'ler tablonun yuzde kacine ugradi? (coverage)
2. Kullanim Zipf mi (az satir cok kullaniliyor) mi dagilmis?
3. Injection sinyali, hidden state'e gore ne kadar guclu? (alpha*|out| / |h|)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from configs.config import EngramExperimentConfig, ModelConfig
from src.data_loader import PackedTokenDataset
from src.engram import EngramAttach, EngramModuleConfig, HashConfig


def main():
    device = "cuda"
    exp = EngramExperimentConfig()

    # --- 1. Tablo doluluk analizi -------------------------------------------
    print("Hash mapping kuruluyor (tokenizer sikistirma ~1 dk)...")
    h_cfg = HashConfig(
        tokenizer_name_or_path=ModelConfig.name,
        layer_ids=exp.layer_ids,
        max_ngram_size=exp.max_ngram_size,
        n_head_per_ngram=exp.n_head_per_ngram,
        vocab_multiplier=exp.vocab_multiplier,
        seed=exp.seed,
    )
    mapping = h_cfg and __import__("src.engram.hashing", fromlist=["NgramHashMapping"]).NgramHashMapping(h_cfg)
    flat_sizes = [p for ngram_primes in mapping.table_sizes for p in ngram_primes]
    n_cols = len(flat_sizes)

    ds = PackedTokenDataset("data/python_1b", val_tokens=10_000_000)

    # Train bolgesinden orneklem: ~8M token
    rng = np.random.default_rng(42)
    n_batches = 1000
    bsz, seq = 4, 512

    seen = [np.zeros(s, dtype=bool) for s in flat_sizes]
    tok_count = 0
    print(f"{n_batches} batch x {bsz}x{seq} = {n_batches*bsz*seq:,} token taraniyor...")
    for i in range(n_batches):
        x, _ = ds.get_batch(bsz, seq, rng, split="train")
        hh = mapping.hash(x)  # {layer: [B,T,C]}
        for lid in exp.layer_ids:
            t = hh[lid].reshape(-1, n_cols).numpy()
            for c in range(n_cols):
                u = np.unique(t[:, c])
                seen[c][u] = True
        tok_count += x.numel()
        if (i + 1) % 250 == 0:
            cov = np.mean([s.mean() for s in seen])
            print(f"  {tok_count:,} token | ort. coverage: %{cov*100:.1f}")

    covs = [s.mean() for s in seen]
    print("\n=== TABLO DOLULUK ===")
    for c, (s, sz) in enumerate(zip(seen, flat_sizes)):
        kind = "2-gram" if c < exp.n_head_per_ngram else "3-gram"
        print(f"  kolon {c} ({kind}, head {c % exp.n_head_per_ngram}): "
              f"{s.sum():,}/{sz:,} = %{s.mean()*100:.2f}")
    print(f"  ORTALAMA COVERAGE: %{np.mean(covs)*100:.2f}")
    print(f"  (orneklem {tok_count:,} token'dan; tam 993M token ile dogrusal olmayan sekilde artar)")

    # --- 2. Checkpoint agirliklarinda egitilen/egitilmeyen satir normlari -----
    ckpt = torch.load("runs/C_python/engram_step2000.pt", map_location="cpu", weights_only=False)
    emb_key = f"{exp.layer_ids[0]}.multi_head_embedding.embedding.weight"
    emb_w = ckpt["injections"][emb_key].float()  # [total_N, dim]

    # paylasilan tablo tek kayitli; offsetleri coz
    offsets = np.concatenate([[0], np.cumsum(flat_sizes)[:-1]])
    trained_norms, random_norms = [], []
    for c, s in enumerate(seen):
        w = emb_w[offsets[c] : offsets[c] + flat_sizes[c]]
        norms = w.norm(dim=1).numpy()
        trained_norms.append(norms[s])
        random_norms.append(norms[~s])
    tn = np.concatenate(trained_norms)
    rn = np.concatenate(random_norms)
    print("\n=== SATIR NORMLERI (drift proxy'si) ===")
    print(f"  dokunulan satirlar : ort {tn.mean():.4f} (std {tn.std():.4f})")
    print(f"  rastgele satirlar  : ort {rn.mean():.4f} (std {rn.std():.4f})")
    print("  -> fark buyukse egitim tabloyu gercekten degistirmis demektir")

    # --- 3. Sinyal gucu ------------------------------------------------------
    print("\n=== SINYAL GUCU (val orneklemi uzerinde) ===")
    model = __import__("transformers", fromlist=["AutoModelForCausalLM"]).AutoModelForCausalLM.from_pretrained(
        ModelConfig.name, dtype=torch.bfloat16
    ).to(device)
    mod_cfg = EngramModuleConfig(n_embed_per_ngram=exp.n_embed_per_ngram)
    attach = EngramAttach(model, h_cfg, mod_cfg, layer_ids=exp.layer_ids)
    attach.injections.load_state_dict(ckpt["injections"])
    attach.enable()
    model.eval()

    with torch.no_grad():
        for lid in exp.layer_ids:
            inj = attach.injections[str(lid)]
            x, _ = ds.get_batch(4, 512, rng, split="val")
            x = x.to(device)
            # katman girisi yok; girdiyi embedding olarak taklit et ve gercek
            # hidden istatistigini modelin ilk katmanindan almak yerine norm oraniyla yetinelim
            hash_ids = inj._hash_to_device(x, lid)
            emb = inj.multi_head_embedding(hash_ids)
            value = inj.value_proj(emb)
            key = inj.norm_key(inj.key_proj(emb))
            # query icin gercek hidden lazim; basit proxy: norm1 istatistigi
            gate_scale = float(inj.alpha.item())
            v_norm = value.float().pow(2).mean().sqrt().item()
            print(f"  katman {lid}: alpha={gate_scale:+.4f} | tipik |value|~{v_norm:.3f} "
                  f"| etkili enjeksiyon ~{abs(gate_scale)*v_norm:.4f}")


if __name__ == "__main__":
    main()
