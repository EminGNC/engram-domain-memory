# ENGRAM PROJESİ — TAM ELLE VERME DOKÜMANI (Handoff)

> Bu doküman, projeye başka bir AI (veya insan) sıfırdan devam edecekse gereken HER ŞEYİ içerir.
> Yazım tarihi: 2026-08-26 sabahı. Konum: C:\Users\ilker\deneme

---

## 1. PROJE NEDİR? (tek paragraf)

Donuk (değiştirilemez) bir hazır LLM'e (Qwen3-0.6B) sonradan, domain'e özel (Python kodu)
"Engram" adlı harici bir n-gram-hafıza modülü takıp SADECE bu modülü eğitmeyi deniyoruz.
Amaç: aynı veriyle klasik LoRA fine-tuning'den daha iyi (veya en azından rekabetçi) sonuç almak
ve kazanımın domain'e özgü olduğunu göstermek. DeepSeek'in 2026 başındaki "Engram" makalesinden
esinlenildi; onlar belleği pre-training sırasında birlikte eğitti, BİZ SONRADAN takıyoruz
(literatürde test edilmemiş senaryo).

## 2. HİPOTEZLER

| # | Hipotez | Durum |
|---|---|---|
| H1 | Aynı veri+bütçede Engram ≥ LoRA | Henüz cevaplanmadı (sağlıklı Python-Engram koşusu eksik) |
| H2 | Kazanım domain-spesifik: ΔPython > ΔGeneral | **KISMEN DOĞRULANDI** (aşağıda) |
| H3 | Kazanç bellek içeriğinden gelir (rastgele tablo kazanmaz) | Test ediliyor — önemli uyarı sinyali var |
| H4 | Bellek çıkarılınca model birebir orijinale döner | ✅ KANITLANDI (α=0 → bit-bit aynı; testlerle) |

## 3. REFERANS MAKALELER

1. **DeepSeek Engram** — arXiv:2601.07372, github.com/deepseek-ai/Engram
   (n-gram hash lookup, O(1); hash şemasını demo kodlarından `engram_demo_v1.py` uyarladık)
2. **Tokenizer-Agnostic Engram** — arXiv:2607.29065 (polynomial hashing; Faz 3 için)
3. **Memory Grafting** — arXiv:2605.20948 (offline bellek inşası)
4. **Cross-Model Memory Transfer** — arXiv:2608.17050 (reader mimarisi referansı)

## 4. MİMARİ (bizim implementasyon)

```
token dizisi ──► CompressedTokenizer (vocab 151K→~107K, normalize+merge)
                       │
                       ▼
        n-gram hash: 2-gram ve 3-gram, XOR şema,
        katman-bazlı rastgele tek-multiplier'lar, head başina ORTAK asal modüller
        (v2: multiplier=24 → kolon başına ~2.57M satır)
                       │
                       ▼
        TEK PAYLAŞILAN multi-head embedding tablosu (~110M param, fp32)
        (4 kolon × ~2.57M satır × 12 dim)
                       │
                       ▼
   her injection katmanında: value_proj + key_proj → sigmoid gate (içerik bazlı)
                            + depthwise ShortConv
                       │
                       ▼
   h_yeni = h_eski + α · out        (α öğrenilen skalar gate)
```

- **Injection noktaları:** katman 1, 4, 7 (28 katmanın erken kısmı), HF forward-pre-hook ile
- **Eğitilebilir:** sadece Engram modülleri (~123M). Backbone %100 donuk (tie_word_embeddings dahil doğru ele alındı)
- **fp32 KURALI:** Eğitilebilir modüller fp32 saklanmalı (bkz. Bug #5 — projenin en önemli dersi)
- **Tak-çıkar garantisi:** α'yı 0'a çekersen model birebir base'e döner (test edildi)

### Dosya haritası

```
src/engram/
  hashing.py    HashConfig, CompressedTokenizer, NgramHashMapping (+hash_torch GPU-side)
  module.py     MultiHeadEmbedding, ShortConv, EngramInjection (alpha gate'li)
  wrapper.py    EngramAttach: HF modele hook'la takma/çıkarma/dondurma/dedupe-param
src/data_loader.py          PackedTokenDataset (memmap shard'lardan random crop)
configs/config.py           EngramExperimentConfig (v2 değerleri)
train/train_engram.py       C/D/E2 koşuları (--freeze-table = E2 modu, --table-lr ayrık LR)
train/train_lora.py         B koşusu (peft, --target-params ile eşit bütçe kalibrasyonu)
eval/eval_fixed.py          PAIRED eval (sabit token blokları) — tek güvenilir karşılaştırma yöntemi
eval/eval_ppl.py            eski random-crop eval (GÜVENİLMEZ, sadece hızlı bakış)
eval/report.py              log parser
scripts/data/prepare_python.py  stream→filtre→tokenize→.bin shard (her dataset için çalışır)
scripts/smoke_test.py       çekirdek modül testleri
scripts/test_attach.py      gerçek modelde attach/detach doğrulamaları
scripts/diagnose_engram.py  tablo doluluk/norm/sinyal teşhisi
scripts/c3_run.ps1          C3 koşusu + eval zinciri (schtasks ile detached çalışır)
runs/                       tüm koşu çıktıları; runs/overnight/ rapor ve loglar
data/python_1b, data/general_1b   8'er shard .bin (int32 packed tokens)
```

## 5. ORTAM (önemli quirklerle)

- Windows 11, RTX 3070 Ti Laptop **8GB VRAM**, Python 3.12
- torch 2.6.0+cu126 (başta CPU-only kurulmuştu!), transformers 5.15.1, peft 0.20.0, datasets 3.2.0
- **WDDM tuzağı:** VRAM taşarsa Windows sessizce RAM kullanır, hata vermez, her şey yavaşlar.
  Kontrol: `Get-Counter '\GPU Adapter Memory(*)\Shared Usage'` → Shared > 1GB ise spill var demektir.
  Çözüm: bsz/seq küçült. Güvenli konfig: **bsz4×512 ≈ 5.4GB** (bf16 engram), fp32 engram için bsz3.
- **Gradient checkpointing şart** (kapalıyken aktivasyonlar ~20GB).
- `expandable_segments` Windows'ta desteklenmiyor (warning zararsız).
- bigcode/starcoderdata GATED → HF login gerekli (token kaydedildi; **rotate edilmeli!**)
- Uzun işler artık `schtasks` ile DETACHED çalışıyor (arka plan shell zincirleri oturum ölünce ölebiliyordu);
  script'ler `Set-Location` ile dizin sabitliyor (schtasks default cwd=System32).

## 6. VERİ

| Set | İçerik | Boyut | Not |
|---|---|---|---|
| data/python_1b | bigcode/starcoderdata python alt kümesi | 1.003B token, 8 shard | starcoderdata zaten benchmark-dekontamine |
| data/general_1b | HuggingFaceFW/fineweb sample/10BT | 1.001B token, 8 shard | ⚠️ Filtre prose'un %75'ini eledi (MAX_AVG_LINE_LEN=120); korpus kısa-satırlı metinlere kayıyor |

Pipeline: streaming (veri inmiyor!) → filtre → Qwen tokenizer (rayon paralel) → int32 packed .bin shard'lar (128M token/shard). Python paketi 19 dk, FineWeb 79 dk sürdü.

Son 10M token her korpusun validasyon bölgesi (eğitim dışı).

## 7. YAPILAN KOŞULAR VE SONUÇLAR

### Paired eval protokolü (TEK güvenilir yöntem)
`eval_fixed.py`: val bölgesinin sonundan 256 sabit pencere (131K token) — TÜM modeller AYNI tokenlarda ölçülür.
Random-crop eval güvenilmezdir (aynı checkpoint 1.58 de 1.75 de çıktı!).

### Python kod bloğu üzerindeki sonuçlar (Δ vs Base)

| Koşu | Ne? | Ne zaman eğitildi | Δ loss | Yorum |
|---|---|---|---|---|
| A — Base | referans | — | 0.0000 (loss 1.7939 / ppl 6.01) | |
| B — LoRA (~113M, r≈180) | baseline | ✅ sağlam | **−0.2085** (ppl 4.88) | güçlü |
| C2 — Python Engram | ana hipotez | ❌ **BUG'LI bf16 kodu** | +0.0016 | GEÇERSİZ koşu |
| E2 — Rastgele donuk tablo + eğitilen reader | kontrol | ✅ fp32-fix'li | **−0.0367** | "rastgele özellik adaptörü" etkisi! |
| D — General Engram (FineWeb) | domain kontrolü | ✅ fp32-fix'li | −0.0199 | |

### Çapraz-domain (genel metin bloğu, base loss 3.5336)

| Model | Δ genel blokta |
|---|---|
| C2py (python belleği) | +0.0051 (zarar yok) |
| **Dgen (general belleği)** | **−0.1132** (kendi domaininde büyük kazanım!) |

### Sonuçların yorumu

1. **fp32 fix'i kanıtladı:** D'de α hareket etti (0.05→0.065→0.059), tablo normu 3.46→3.195 drift etti,
   ve kendi domaininde −0.113 nat kazandı → **post-hoc Engram ÖĞRENEBİLİYOR**. Yöntem ölüyor değil!
2. **H2 destekleniyor:** D kendi domaininde −0.113 kazanırken yabancı domaine (python) sadece −0.020 taşıyor.
   Bellek domain'e özgü içerik öğreniyor.
3. **⚠️ H3 alarmı:** E2 (rastgele tablo!) python'da −0.0367 kazandı — rastgele özellik adaptörü etkisi.
   Gerçek python-Engram (C3) bunu GEÇMELİ ki "içerik iş yapıyor" diyebilelim.
4. **C2 geçersiz:** bf16 bug'ı döneminde eğitildi; tablosu hiç güncellenmedi, α init'te kaldı.
   Sabah raporundaki "tavana ulaşıldı" kararı bu yüzden HATALI.

## 8. YAKALANAN BUG'LAR (hepsi ders dolu — tekrarlanmamalı!)

1. **Çift label-shift:** transformers `labels`'ı zaten kaydırır. `labels=y` (önceden kaydırılmış) vermek
   loss'u 13.0 gösteriyordu (gerçek 1.67). Doğrusu: `model(input_ids=x, labels=x)`.
2. **Katman-başına farklı hash asalları:** paylaşılan tablo katman1 boyutundaydı → katman 4/7 OOB
   → CUDA device-side assert (CUBLAS hatası diye maskelenmişti!). Fix: asallar tüm katmanlarda ortak.
3. **Paylaşılan parametrenin optimizer'a 3 kez girmesi:** ModuleDict içindeki paylaşımılmış tabloyu
   `named_parameters()` her injection'dan birer kez verir → dedupe şart (`wrapper.trainable_parameters`),
   yoksa etkili LR 3 katına çıkar.
4. **CPU-side hashing:** numpy hash her adımda CPU'da → GPU'da `hash_torch()` yazıldı (~%35 kazanç).
5. **WDDM RAM-spill:** 16GB kullanım = sessiz hız katliamı (400 tok/s). Saf VRAM'de 2030 tok/s.
6. **peft + GC sıralaması:** `gradient_checkpointing_enable()` peft sarmalamasından ÖNCE çağrılırsa etkisiz kalıyor.
7. **GC sadece training modunda:** `from_pretrained()` modeli eval() yükler; `model.train()` unutulursa
   GC sessizce kapalı kalır → RAM spill. (LoRA script'inde oldu.)
8. **★ EN ÖNEMLİSİ — bf16 parametre yuvarlaması:** Eğitilebilir Engram parametreleri bf16'dayken
   AdamW adımları (lr~1e-4) çoğu elemanda ULP'nin altında kalıp YUVARLANIP KAYBOLUYORDU.
   Kanıt: α ve tablo 6000 adımda bit-bit aynı kaldı. Fix: eğitilebilir modüller fp32,
   injection çıktısı `.to(hidden.dtype)` ile residual akışa geri giriyor.
   Bu bug C-v1/C-v2'nin "öğrenememesinin" gerçek sebebiydi — mimari suçlu değilmiş.
9. **Ölü-gate dinamiği:** α=0 init → tablo gradyanı ∝ α → tablo öğrenemez → α büyüyemez (kısır döngü).
   v2 fix: α_init=0.05 + tabloya ayrık yüksek LR (5e-4). (Ama asıl kurtarıcı #8 idi.)
10. **Hash collision felaketi (v1):** multiplier=2 iken kolon başına ~215K satır → satıra ~50K farklı
    n-gram çarpıyordu. v2: multiplier=24 → collision ~12× azaldı.
11. **schtasks cwd:** zamanlanmış görev System32'de başlar → script başına `Set-Location` ekle.

## 9. ŞU ANKİ DURUM (2026-08-26 sabahı)

- Kullanıcı C3'ü durdurdu ("şuan yapamam") — eğitim step ~2000 civarındayken kesildi, yarım checkpoint var.
- GPU boşta, hiçbir süreç çalışmıyor. Zamanlanmış görevler iptal.
- **Bekleyen asıl soru:** fp32-fix'li Python-Engram (C3), rastgele-tablo kontrolünü (E2: −0.0367)
  geçebilecek mi? Geçerse H3 ✓ ve H1'in gerçek cevabı gelir.

## 10. DEVAM ETMEK İSTERSEN

```powershell
# C3'ü baştan başlat (85 dk) — ya da schtasks /run /tn "EngramC3"
python train/train_engram.py --data-dir data/python_1b --steps 8000 --bsz 4 --seq-len 512 `
    --eval-interval 2000 --alpha-init 0.05 --table-lr 5e-4 --out-dir runs/C3_python

# Ardından paired eval:
python eval/eval_fixed.py --lora runs/B_lora/adapter_step2000 `
    --engrams @("E2=runs/E2_random/engram_step6000.pt","D=runs/D_general/engram_step8000.pt","C3=runs/C3_python/engram_step8000.pt")
```

Karar ağacı:
- ΔC3 < −0.04 (E2'yi açık ara geçerse): hipotez canlı → uzun koşu (50M+ token) + hiperparametre taraması
- ΔC3 ∈ (−0.037, 0): kazanım sadece rastgele-özellik adaptöründen geliyor → içerik hipotezi zayıf;
  hikayeyi H2 (D'nin −0.11'i) + H4 (portability) üzerine kur
- ΔC3 > 0: python'da headroom yok demektir → domain değiştir (base'in zayıf olduğu niş alan)

## 11. DAHA FAZLA HESAP GÜCÜ İÇİN (kullanıcının sorusu üstüne)

Laptop yerine bulut: veri HF'den stream edildiği için TAŞINACAK KOD YOK (küçük repo).
Seçenekler: Colab (ücretsiz T4, kopma riskli), Kaggle (9-12 saat oturum, ücretsiz),
Vast.ai/RunPod (RTX 3090 ~$0.2-0.4/saat — tüm matris $5-15'e biter; SSH+tmux en dayanıklısı).
Taşıma: git clone + pip install + hf login + prepare script'leri (~2 saat hazırlık) + run_all zinciri.

## 12. KÜÇÜK AMA ÖNEMLİ NOTLAR

- HF token chata yapıştırmıştı → işler stabilleşince ROTATE edilmeli.
- `PROJE_OZETI.md` eski durumu yansıtıyor (C-v2 öncesi); bu doküman daha güncel.
- Plan dokümanları: `C:\Users\ilker\.opencode\plan\engram-*.md`
- Checkpoint'ler Engram state_dict'i içerir (paylaşılan tablo torch.save sayesinde tek storage).
- Disk: ~90 GB boştu; veri 8 GB + checkpoint'ler koşu başına ~2.5GB.
