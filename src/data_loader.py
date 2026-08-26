"""Paketli .bin token shard'lari uzerinde memmap tabanli veri yukleyici.

- Tum shard'lar tek mantiksal token akisi gibi davranir
- Son `val_tokens` kadar token validasyona ayrilir (egitim disi)
- get_batch: rastgele offset'lerden [B, L+1] okuyup x/y ciftine boler
"""

import bisect
from pathlib import Path

import numpy as np
import torch


class PackedTokenDataset:
    def __init__(self, data_dir: str, val_tokens: int = 0):
        self.files = sorted(Path(data_dir).glob("*.bin"))
        if not self.files:
            raise FileNotFoundError(f"{data_dir} altinda .bin yok")
        self.maps = [np.memmap(f, dtype=np.int32, mode="r") for f in self.files]
        self.sizes = [len(m) for m in self.maps]
        self.cum = np.cumsum(self.sizes).tolist()
        self.total = self.cum[-1]
        if val_tokens >= self.total:
            raise ValueError("val_tokens toplamdan buyuk")
        self.train_end = self.total - val_tokens

    def _read_span(self, start: int, length: int) -> np.ndarray:
        out = np.empty(length, dtype=np.int32)
        pos = 0
        while pos < length:
            fi = bisect.bisect_right(self.cum, start + pos)
            local = (start + pos) - (self.cum[fi - 1] if fi > 0 else 0)
            take = min(length - pos, self.sizes[fi] - local)
            out[pos : pos + take] = self.maps[fi][local : local + take]
            pos += take
        return out

    def get_batch(self, batch_size: int, seq_len: int, rng: np.random.Generator, split: str = "train") -> tuple:
        end = self.train_end if split == "train" else self.total
        max_start = end - seq_len - 1
        starts = rng.integers(0, max_start, size=batch_size)
        x = torch.empty((batch_size, seq_len), dtype=torch.long)
        y = torch.empty((batch_size, seq_len), dtype=torch.long)
        for i, s in enumerate(starts):
            span = torch.from_numpy(self._read_span(int(s), seq_len + 1).astype(np.int64))
            x[i], y[i] = span[:-1], span[1:]
        return x, y
