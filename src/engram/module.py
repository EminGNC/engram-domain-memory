"""Engram modulu: multi-head embedding + gated residual injection.

DeepSeek Engram demosundan uyarlanmistir, su farklarla:
- Hyper-connection (hc_mult) yok; Qwen3 duz [B, T, D] hidden state kullanir
- Zero-init learnable alpha gate eklendi: baslangicta model birebir korunur (tak-cikar guvencesi)
- Tum boyutlar config'ten gelir
"""

import math
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn

from .hashing import HashConfig, NgramHashMapping


@dataclass
class EngramModuleConfig:
    n_embed_per_ngram: int = 256       # n-gram embedding toplam boyutu
    n_head_per_ngram: int = 4          # hashing config ile ayni olmali
    kernel_size: int = 4               # short conv genisligi
    gate_temperature: float = 1.0
    alpha_init: float = 0.0            # v2'de 0.05: olu-gate dongusunu kirmak icin


class MultiHeadEmbedding(nn.Module):
    """Head'leri tek buyuk Embedding tablosunda tutar; head offset'leri ile ayirir."""

    def __init__(self, list_of_n: List[int], dim: int):
        super().__init__()
        offsets = [0]
        for n in list_of_n[:-1]:
            offsets.append(offsets[-1] + n)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))

        self.embedding = nn.Embedding(sum(list_of_n), dim)

    def forward(self, hash_ids: torch.Tensor) -> torch.Tensor:
        """hash_ids: [..., num_heads] -> [..., num_heads * dim]"""
        return self.embedding(hash_ids + self.offsets).flatten(start_dim=-2)


class ShortConv(nn.Module):
    """Demo'daki depthwise short conv'un sadelestirilmis hali (hyper-connection yok)."""

    def __init__(self, hidden_size: int, kernel_size: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=False,
            padding=kernel_size - 1,
        )
        self.norm = nn.RMSNorm(hidden_size)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, D] -> [B, T, D]"""
        y = self.conv(self.norm(x).transpose(1, 2))
        return self.act(y[..., : x.size(1)]).transpose(1, 2)


class EngramInjection(nn.Module):
    """Tek bir transformer katmanina baglanan Engram enjeksiyon modulu.

    forward(hidden_states, input_ids) -> engram cikisi.
    Kullanim: h_yeni = h_eski + alpha * engram_out  (alpha zero-init)
    """

    def __init__(
        self,
        hash_cfg: HashConfig,
        module_cfg: EngramModuleConfig,
        hidden_size: int,
        shared_embedding: "MultiHeadEmbedding" = None,
    ):
        super().__init__()
        assert module_cfg.n_head_per_ngram == hash_cfg.n_head_per_ngram

        self.hash_mapping = NgramHashMapping(hash_cfg)
        head_dim = module_cfg.n_embed_per_ngram // module_cfg.n_head_per_ngram
        if shared_embedding is not None:
            # Katmanlar arasi PAYLASILAN tablo (parametre butcesi icin kritik)
            self.multi_head_embedding = shared_embedding
        else:
            flat_sizes = [p for ngram_primes in self.hash_mapping.table_sizes for p in ngram_primes]
            self.multi_head_embedding = MultiHeadEmbedding(list_of_n=flat_sizes, dim=head_dim)
        engram_dim = (hash_cfg.max_ngram_size - 1) * module_cfg.n_embed_per_ngram

        self.value_proj = nn.Linear(engram_dim, hidden_size)
        self.key_proj = nn.Linear(engram_dim, hidden_size)
        self.norm_key = nn.RMSNorm(hidden_size)
        self.norm_query = nn.RMSNorm(hidden_size)
        self.short_conv = ShortConv(hidden_size, kernel_size=module_cfg.kernel_size)

        # Dis gate: v1'de zero-init idi (tak-cikar guvencesi); bu, tablo gradyanini
        # boğan "olu-gate" dongusu yaratti. v2'de kucuk pozitif init kullaniliyor.
        self.alpha = nn.Parameter(torch.full((1,), float(module_cfg.alpha_init)))

    @torch.no_grad()
    def _hash_to_device(self, input_ids: torch.Tensor, layer_id: int) -> torch.Tensor:
        # GPU-side hashing (numpy/CPU yolu yavastı; bkz. NgramHashMapping.hash_torch)
        return self.hash_mapping.hash_torch(input_ids)[layer_id]

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor, layer_id: int) -> torch.Tensor:
        """
        hidden_states: [B, T, D]
        input_ids:     [B, T]
        """
        hash_ids = self._hash_to_device(input_ids, layer_id)
        emb = self.multi_head_embedding(hash_ids)  # [B, T, engram_dim]

        value = self.value_proj(emb)  # [B, T, D]
        key = self.norm_key(self.key_proj(emb))
        query = self.norm_query(hidden_states)

        # Icerik-bazli sigmoid gate: sorgu ile anahtar uyusursa acilir
        gate = torch.sigmoid((key * query).sum(-1, keepdim=True) / math.sqrt(hidden_states.size(-1)))
        out = gate * value + self.short_conv(gate * value)

        # fp32 hesap -> backbone'un bf16 akisina geri cast ederek ekle
        return (self.alpha * out).to(hidden_states.dtype)

    def trainable_parameters(self):
        """Backbone donuk; sadece bu parametreler egitilir."""
        return self.parameters()
