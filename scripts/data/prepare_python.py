"""Python kod corpusu hazirlama: HF stream -> filtre -> tokenize -> paketli .bin shard.

Strateji (disk dostu):
- Veriseti INDIRILMEZ; datasets streaming ile sirali okunur
- Filtrelerden gecen dokumanlar Qwen tokenizer ile token'lara cevrilir
- Token'lar kesintisiz tek akisa paketlenir, sabit boyutlu .bin shard'lara yazilir (int32)
- Egitimde memmap + rastgele crop ile okunur (bkz. train/)

Kullanim ornegi:
    python scripts/data/prepare_python.py --target-tokens 1000000000 --out-dir data/python_1b

Gereksinimler:
- bigcode/starcoderdata GATED'tir: HF hesabiyla lisansi kabul edip access token uretmelisin
  https://huggingface.co/datasets/bigcode/starcoderdata
  Token'i ya --hf-token ile ver ya da once `huggingface-cli login` yap.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# --- Varsayilanlar -----------------------------------------------------------

DEFAULT_DATASET = "bigcode/starcoderdata"
DEFAULT_CONFIG = "python"

MIN_BYTES = 200        # cok kucuk dosyalar engram icin bilgi tasimaz
MAX_BYTES = 80_000     # cok buyuk dosyalar (minified/data blob) elenir
MAX_AVG_LINE_LEN = 120 # minified kod tespiti
MIN_ALNUM_RATIO = 0.25 # ikili/dump dosya tespiti


def passes_filters(text: str) -> bool:
    n = len(text)
    if n < MIN_BYTES or n > MAX_BYTES:
        return False
    lines = text.splitlines()
    if not lines:
        return False
    avg_line = sum(len(l) for l in lines) / len(lines)
    if avg_line > MAX_AVG_LINE_LEN:
        return False
    alnum = sum(c.isalnum() for c in text) / n
    if alnum < MIN_ALNUM_RATIO:
        return False
    return True


class ShardWriter:
    """Token akisini sabit boyutlu int32 .bin shard'lara yazar."""

    def __init__(self, out_dir: Path, shard_tokens: int = 128_000_000):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_tokens = shard_tokens
        self.shard_idx = 0
        self._buf = np.empty(shard_tokens, dtype=np.int32)
        self._pos = 0
        self.total_tokens = 0

    def add(self, tokens: np.ndarray) -> None:
        tokens = tokens.astype(np.int32, copy=False)
        src = 0
        while src < len(tokens):
            take = min(len(tokens) - src, self.shard_tokens - self._pos)
            self._buf[self._pos : self._pos + take] = tokens[src : src + take]
            self._pos += take
            src += take
            self.total_tokens += take
            if self._pos == self.shard_tokens:
                self.flush()

    def flush(self) -> None:
        if self._pos == 0:
            return
        path = self.out_dir / f"shard_{self.shard_idx:05d}.bin"
        self._buf[: self._pos].tofile(path)
        print(f"  shard yazildi: {path.name} ({self._pos:,} tokens)")
        self.shard_idx += 1
        self._pos = 0

    @property
    def full(self) -> bool:
        return False  # disaridan stop kontrol edilir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--target-tokens", type=int, default=1_000_000_000)
    ap.add_argument("--out-dir", default="data/python_1b")
    ap.add_argument("--hf-token", default=None)
    ap.add_argument("--max-docs", type=int, default=None, help="guvenlik siniri")
    ap.add_argument("--content-key", default=None,
                    help="dokuman metin alani (starcoderdata: content/code, fineweb: text)")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"Tokenizer yukleniyor...")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    print(f"Stream aciliyor: {args.dataset} [{args.config}]")
    ds = load_dataset(
        args.dataset,
        data_files=f"{args.config}/*.parquet",
        split="train",
        streaming=True,
        token=args.hf_token,
    )

    writer = ShardWriter(Path(args.out_dir))
    doc_buffer: list[str] = []
    buffered_chars = 0
    n_seen = n_kept = 0
    t0 = time.time()
    next_report = 500

    def flush_buffer():
        nonlocal doc_buffer, buffered_chars
        if not doc_buffer:
            return
        # encode_batch rayon ile paralellesir (TOKENIZERS_PARALLELISM=true gerekli)
        enc = tok._tokenizer.encode_batch(doc_buffer, add_special_tokens=False)
        for encoding in enc:
            if len(encoding.ids) > 1:
                writer.add(np.asarray(encoding.ids))
        doc_buffer = []
        buffered_chars = 0

    try:
        for ex in ds:
            n_seen += 1
            if args.content_key:
                text = ex.get(args.content_key)
            else:
                content_key = "content" if "content" in ex else ("code" if "code" in ex else "text")
                text = ex.get(content_key)
            if text and passes_filters(text):
                n_kept += 1
                doc_buffer.append(text)
                buffered_chars += len(text)
                if buffered_chars > 16_000_000:  # ~16MB'de bir tokenize et (paralel verimliligi icin)
                    flush_buffer()
            if writer.total_tokens >= args.target_tokens:
                break
            if args.max_docs and n_seen >= args.max_docs:
                break
            if n_seen % next_report == 0:
                rate = writer.total_tokens / max(time.time() - t0, 1)
                eta_min = (args.target_tokens - writer.total_tokens) / max(rate, 1) / 60
                print(
                    f"  {n_seen:,} dokuman tarandi | {n_kept:,} tutuldu | "
                    f"{writer.total_tokens:,}/{args.target_tokens:,} tokens | "
                    f"{rate:,.0f} tok/s | ETA ~{eta_min:.0f} dk"
                )
                next_report *= 2
    except KeyboardInterrupt:
        print("\nKesildi (Ctrl+C), buffer flush ediliyor...")

    flush_buffer()
    writer.flush()

    dt = time.time() - t0
    print(
        f"\nBitti: {writer.total_tokens:,} tokens | {n_kept:,}/{n_seen:,} dokuman | {dt/60:.1f} dk"
    )
    print(f"Cikti: {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()
