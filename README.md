# Domain-Spesifik Engram

Donuk, Engram'sız ön-eğitimli bir LLM'e (pilot: Qwen3-0.6B) sonradan domain-spesifik
(Python) Engram belleği eklenip eklenemeyeceğini ve bunun LoRA'dan daha iyi olup
olmadığını test eden deney projesi.


## Yapı

```
src/engram/
  hashing.py    # CompressedTokenizer + NgramHashMapping (DeepSeek demosundan uyarlanmış)
  module.py     # MultiHeadEmbedding + EngramInjection (zero-init gated residual)
  wrapper.py    # HF modeline hook ile takma/çıkarma
configs/
  config.py     # Tüm deney hiperparametreleri
scripts/
  smoke_test.py # Hash + injection ileri geçiş testi (model ağırlığı gerektirmez)
train/          # Eğitim betikleri (Faz 1)
eval/           # Değerlendirme (perplexity, HumanEval)
```

## Kurulum

```bash
pip install torch==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install "transformers>=4.51" datasets sympy
python scripts/smoke_test.py
```

## Referanslar

- DeepSeek Engram: arXiv:2601.07372 / github.com/deepseek-ai/Engram
- Tokenizer-Agnostic Engram: arXiv:2607.29065
- Memory Grafting: arXiv:2605.20948
- Cross-Model Memory Transfer: arXiv:2608.17050
