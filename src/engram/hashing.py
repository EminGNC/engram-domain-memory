"""Engram hashing: compressed tokenizer + n-gram hash mapping.

DeepSeek Engram demosundan (engram_demo_v1.py) uyarlanmistir:
- XOR tabanli coklu-head hashing, katman basina ayri multiplier'lar
- Basinc icin tokenizer vokabulerisi normalize edilerek sikistirilir

Farklar (bu proje):
- Tum parametreler disaridan EngramConfig ile verilir (global yok)
- pad_id otomatik cozulur
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
from sympy import isprime
from tokenizers import Regex, normalizers
from transformers import AutoTokenizer


@dataclass
class HashConfig:
    tokenizer_name_or_path: str = "Qwen/Qwen3-0.6B"
    # Her n-gram derecesi icin hash taban boyutu (head sayisiyla carpilacak)
    vocab_multiplier: int = 2  # demo'da 5 idi; tablo butcesini kucuk tutmak icin 2
    max_ngram_size: int = 3  # 2-gram ve 3-gram kullanilir
    n_head_per_ngram: int = 4
    layer_ids: List[int] = field(default_factory=lambda: [1, 4, 7])
    seed: int = 0


class CompressedTokenizer:
    """Tokenizer vokabulerisini normalize ederek sikistirir.

    Ayni normalize edilmete dusuren token id'leri tek "sikistirilmis" id'e maplenir.
    Boylece hash uzayi kuculur ve (ileride) tokenizer-agnostik paylasima zemin hazirlanir.
    """

    def __init__(self, tokenizer_name_or_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, trust_remote_code=True
        )

        sentinel = "\ue000"
        self.normalizer = normalizers.Sequence(
            [
                normalizers.NFKC(),
                normalizers.NFD(),
                normalizers.StripAccents(),
                normalizers.Lowercase(),
                normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
                normalizers.Replace(Regex(r"^ $"), sentinel),
                normalizers.Strip(),
                normalizers.Replace(sentinel, " "),
            ]
        )

        self.lookup_table, self.num_new_token = self._build_lookup_table()

    def __len__(self):
        return self.num_new_token

    def _build_lookup_table(self):
        old2new = {}
        key2new = {}
        new_tokens = []

        vocab_size = len(self.tokenizer)
        for tid in range(vocab_size):
            text = self.tokenizer.decode([tid], skip_special_tokens=False)

            if "\ufffd" in text:
                # decode edilemeyen byte token -> raw token string'i kullan
                key = self.tokenizer.convert_ids_to_tokens(tid)
            else:
                norm = self.normalizer.normalize_str(text)
                key = norm if norm else text

            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid

        lookup = np.empty(vocab_size, dtype=np.int64)
        for tid in range(vocab_size):
            lookup[tid] = old2new[tid]

        return lookup, len(new_tokens)

    def compress(self, input_ids: np.ndarray) -> np.ndarray:
        arr = np.asarray(input_ids, dtype=np.int64)
        out = arr.copy()
        valid = arr >= 0  # -1 (pad) degerlerine dokunma
        out[valid] = self.lookup_table[arr[valid]]
        return out

    def __call__(self, input_ids):
        return self.compress(input_ids)


def _find_next_prime(start: int, seen_primes: set) -> int:
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1


class NgramHashMapping:
    """Token id dizisini, katman basina n-gram hash id matrislerine cevirir.

    Cikti boyutu: [B, T, (max_ngram_size - 1) * n_head_per_ngram]
      - ilk n_head kolonlar 2-gram, sonraki n_head kolonlar 3-gram hash'leri
    """

    def __init__(self, cfg: HashConfig):
        self.cfg = cfg

        self.compressed_tokenizer = CompressedTokenizer(cfg.tokenizer_name_or_path)
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)

        # Qwen pad token'i (<|endoftext|>) sikistirilmis uzaya maplenir
        eos_id = self.compressed_tokenizer.tokenizer.eos_token_id or 0
        self.pad_id = int(self.compressed_tokenizer.lookup_table[eos_id])

        # Katman basina deterministik rastgele tek-multiplier'lar
        prime_1 = 10007
        max_long = np.iinfo(np.int64).max
        half_bound = max(1, int(max_long // self.tokenizer_vocab_size) // 2)

        self.layer_multipliers: Dict[int, np.ndarray] = {}
        for layer_id in cfg.layer_ids:
            g = np.random.default_rng(int(cfg.seed + prime_1 * int(layer_id)))
            r = g.integers(low=0, high=half_bound, size=(cfg.max_ngram_size,), dtype=np.int64)
            self.layer_multipliers[layer_id] = r * 2 + 1  # tek sayilar (xor karismasi icin)

        # Head basina asal moduller — TUM KATMANLARDA ORTAK.
        # Katmanlar arasi ayrisim multiplier'lardan gelir; tablo boylece paylasilabilir.
        # (Orijinal demoda her katmanin ayri tablosu vardi; burada tek tablo var.)
        self.head_primes: Dict[int, List[List[int]]] = {}
        seen_primes: set = set()
        base = self.tokenizer_vocab_size * cfg.vocab_multiplier
        shared: List[List[int]] = []
        start = base
        for _ in range(cfg.max_ngram_size - 1):  # 2..max_ngram
            heads = []
            for _ in range(cfg.n_head_per_ngram):
                p = _find_next_prime(start, seen_primes)
                seen_primes.add(p)
                heads.append(p)
                start = p
            shared.append(heads)
        for layer_id in cfg.layer_ids:
            self.head_primes[layer_id] = shared

    @property
    def table_sizes(self) -> List[List[int]]:
        """Engram embedding tablosunun head basina satir sayilari (tum katmanlar ortak)."""
        first = self.cfg.layer_ids[0]
        return self.head_primes[first]

    def _load_gpu_tensors(self, device: torch.device):
        """Hash hesabini GPU'da yapabilmek icin lookup tablosunu ve multiplier'lari tasi."""
        self._gpu_lookup = torch.from_numpy(self.compressed_tokenizer.lookup_table).to(device)
        self._gpu_multipliers = {
            lid: torch.from_numpy(m).to(device) for lid, m in self.layer_multipliers.items()
        }
        self._gpu_device = device

    def hash_torch(self, input_ids: torch.Tensor) -> Dict[int, torch.Tensor]:
        """GPU'da hash: input_ids [B, T] -> {layer_id: [B, T, H] long}"""
        if not hasattr(self, "_gpu_lookup"):
            self._load_gpu_tensors(input_ids.device)

        comp = self._gpu_lookup[input_ids.clamp_min(0)]  # negatif (pad) -> 0
        B, T = comp.shape

        all_layers = {}
        for layer_id in self.cfg.layer_ids:
            mult = self._gpu_multipliers[layer_id]

            shifts = [
                torch.nn.functional.pad(comp, (k, 0), value=self.pad_id)[:, :T]
                for k in range(self.cfg.max_ngram_size)
            ]
            hashes = []
            for n in range(2, self.cfg.max_ngram_size + 1):
                mix = shifts[0] * mult[0]
                for k in range(1, n):
                    mix = mix ^ (shifts[k] * mult[k])
                primes = self.head_primes[layer_id][n - 2]
                for p in primes:
                    hashes.append(mix % p)
            all_layers[layer_id] = torch.stack(hashes, dim=2)
        return all_layers

    def _hash_single_layer(self, compressed_ids: np.ndarray, layer_id: int) -> np.ndarray:
        x = compressed_ids
        B, T = x.shape
        multipliers = self.layer_multipliers[layer_id]

        def shift_k(k: int) -> np.ndarray:
            if k == 0:
                return x
            return np.pad(
                x, ((0, 0), (k, 0)), mode="constant", constant_values=self.pad_id
            )[:, :T]

        shifts = [shift_k(k) for k in range(self.cfg.max_ngram_size)]
        all_hashes = []

        for n in range(2, self.cfg.max_ngram_size + 1):
            mix = shifts[0] * multipliers[0]
            for k in range(1, n):
                mix = np.bitwise_xor(mix, shifts[k] * multipliers[k])
            primes = self.head_primes[layer_id][n - 2]
            for p in primes:
                all_hashes.append((mix % p).astype(np.int64, copy=False))

        return np.stack(all_hashes, axis=2)  # [B, T, num_heads_total]

    def hash(self, input_ids) -> Dict[int, torch.Tensor]:
        """input_ids: [B, T] tensor/list -> {layer_id: [B, T, H] long tensor}"""
        ids = input_ids.detach().cpu().numpy() if torch.is_tensor(input_ids) else np.asarray(input_ids)
        compressed = self.compressed_tokenizer(ids)
        return {
            layer_id: torch.from_numpy(self._hash_single_layer(compressed, layer_id))
            for layer_id in self.cfg.layer_ids
        }
