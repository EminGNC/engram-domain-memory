# ENGRAM PROJESİ — TESTLER VE SONUÇLAR

> Bu doküman, projede yapılan tüm testleri, testlerin neyi doğruladığını ve elde edilen
> sayısal sonuçları içerir. Mimari/arka plan için `HANDOFF.md`'ye bakılır.
> Güncelleme: 2026-08-26 — C3 koşusu Kaggle T4'te başladı (prep fazında).

---

## BÖLÜM 1 — DOĞRULAMA TESTLERİ (birim/smoke seviyesi)

Bu testler "sistem doğru mu kurulmuş" sorusunu cevaplar. Hepsi geçti.

### T1. Çekirdek modül smoke testi (`scripts/smoke_test.py`)

| # | Test | Beklenen | Sonuç |
|---|---|---|---|
| 1 | Hash → embedding → injection boyut zinciri | `[B,T,1024]` çıkış | ✅ |
| 2 | **α=0 iken çıktı birebir sıfır** | modül = kimlik | ✅ (tak-çıkar güvencesi) |
| 3 | α=0.5 iken loss oluşur ve gradyan akar | 10/10 parametrede gradyan | ✅ |
| 4 | Tablo + projeksiyon parametre sayısı | ~100M hedefine yakın | ✅ 110M + 1.1M |

**Neyi kanıtlar:** Engram modülü matematiksel olarak tanımlandığı gibi davranıyor;
"bellek takılı ama kapalıyken model değişmiyor" garantisi ilk günden test ediliyor.

### T2. Gerçek model attach/detach testleri (`scripts/test_attach.py`, GPU)

Qwen3-0.6B ağırlıklarıyla, gerçek forward pass üzerinde:

| # | Test | Sonuç |
|---|---|---|
| 1 | `attach` + α=0 → logits base ile **bit-bit aynı** | ✅ fark: 0.0 |
| 2 | `disable()` → yine birebir aynı | ✅ fark: 0.0 |
| 3 | Loss → embedding tablosuna gradyan akıyor | ✅ |
| 4 | Loss → α gate'e gradyan akıyor | ✅ |
| 5 | Eğitilebilir parametreler dedupe edilmiş (paylaşılan tablo 1 kez) | ✅ 113M |
| 6 | Backbone tamamen donuk (`requires_grad=False`) | ✅ |
| 7 | `remove()` → hook'suz orijinale dönüş | ✅ fark: 0.0 |

**Neyi kanıtlar:** H4 hipotezi (tak-çıkar güvenliği) teknik düzeyde doğrulandı.
Belleği sökünce modelin orijinal davranışına bit-bit dönmesi garanti.

### T3. Teşhis testleri (`scripts/diagnose_engram.py`)

C-v1 checkpoint'i üzerinde çalıştırıldı; "neden öğrenme zayıf?" sorusunu cevapladı:

| Ölçüm | Değer | Yorum |
|---|---|---|
| Tablo coverage (2M token örnekleminde) | %99.93 | Adres uzayı çok küçüktü → satıra ~50K n-gram collision |
| Dokunulan satır normu | 7.969 (init: 8.000) | v1'de tablo neredeyse hiç değişmemiş |
| α değerleri (v1) | ±0.02–0.03 | Sinyal gücü ihmal edilebilir |

**Çıkarılan düzeltmeler:** multiplier 2→24, kolon 8→4, head_dim 64→12 (iso-param),
α_init=0.05, tabloya ayrık LR (5e-4).

---

## BÖLÜM 2 — BUG AVI (testlerin yakaladığı hatalar)

Her bug bir test/teşhis tarafından yakalandı; fix'ler kalıcı:

| # | Bug | Belirti | Nasıl yakalandı | Fix |
|---|---|---|---|---|
| 1 | Çift label-shift | Loss 13.0 (gerçek 1.67) | Metin direkt encode edip karşılaştırma | `labels=x` kullanımı |
| 2 | Katman-başına farklı hash asalları | CUDA device-side assert | CUDA_LAUNCH_BLOCKING ile satır yakalama | Ortak asal modüller |
| 3 | Paylaşılan tablo optimizer'da 3× | Etkili LR 3 katı | Parametre sayımı uyuşmazlığı | ID-based dedupe |
| 4 | CPU-side hashing | ~%35 yavaşlık | Profil | `hash_torch()` GPU'da |
| 5 | WDDM RAM-spill | 400 tok/s duvarı | `GPU Adapter Memory\Shared Usage` sayacı | Konfig küçültme (bsz4×512) |
| 6 | peft GC iptali | LoRA koşusu 10GB RAM kullandı | Shared Usage tekrar | GC'yi sarmalamadan SONRA çağır |
| 7 | eval-mode'da GC kapalı | (6'nın gerçek kökü) | model.train() eksikliği tespiti | `model.train()` eklendi |
| 8 | **bf16 parametre yuvarlaması** | α ve tablo 6000 adımda bit-bit aynı | Checkpoint forensics | **Eğitilebilir modüller fp32** |
| 9 | Ölü-gate kısır döngüsü | α=0 init → tablo açlık | Teşhis T3 | α_init=0.05 |
| 10 | Hash collision felaketi | Coverage %99.93 + norm drift yok | Teşhis T3 | multiplier 24, iso-param yeniden dengeleme |
| 11 | `.gitignore` alt-dizin tuzağı | `scripts/data/` repoya hiç girmedi | Kaggle'da FileNotFoundError | `/data/` kök-anchor |

---

## BÖLÜM 3 — DENEYSEL SONUÇLAR

### 3.1 Değerlendirme protokolü notu

İlk denemelerde rastgele-crop eval kullanıldı; AYNI checkpoint farklı seed'lerde
1.58 ve 1.75 gibi farklı sonuçlar verdi (val bölgesinin zor/kolay bölgeleri dengesiz örnekleniyor).
Bu yüzden **paired eval** yazıldı: val bölgesinin sonundan 256 sabit pencere (131K token),
tüm modeller aynı tokenlarda ölçülür. Aşağıdaki tüm sonuçlar paired protokolle.

### 3.2 Laptop (RTX 3070 Ti) koşuları — python kod bloğu

Base loss: **1.7939** (ppl 6.01)

| Koşu | Eğitim verisi | Trainable | Δ loss | Δ ppl | Geçerlilik |
|---|---|---|---|---|---|
| B — LoRA | Python | ~113M | **−0.2085** | −1.13 | ✅ sağlam |
| C2 — Python Engram | Python | ~123M | +0.0016 | +0.01 | ❌ **GEÇERSİZ** (bf16 bug'lı dönem) |
| E2 — Rastgele donuk tablo + eğitilen reader | Python | ~3M (reader) | **−0.0367** | −0.21 | ✅ fp32-fix sonrası |
| D — General Engram | FineWeb | ~123M | −0.0199 | −0.12 | ✅ fp32-fix sonrası |

### 3.3 Çapraz-domain sonuçlar (genel metin bloğu)

Base loss (genel blok): **3.5336** (ppl 34.25)

| Model | Δ genel blokta | Yorum |
|---|---|---|
| C2py (python belleği) | +0.0051 | Zararsız, nötr |
| **Dgen (general belleği)** | **−0.1132** | Kendi domaininde büyük kazanım |

### 3.4 Bu tablolardan çıkan bilimsel bulgular

**Bulgu 1 — Post-hoc Engram ÖĞRENİYOR (fp32 fix sonrası):**
D-general kendi domaininde −0.113 nat kazandı; α (0.05→0.065→0.059) ve tablo
normları (3.46→3.20) sağlıklı drift etti. Yöntem prensipte çalışıyor.
Önceki "öğrenememe"nin sebebi mimari değil, **bf16 yuvarlama bug'ıydı** (Bug #8).

**Bulgu 2 — Domain specificity sinyali (H2 ✓ yönlü):**
D-general kendi bloğunda −0.113 kazanırken python bloğunda sadece −0.020'ye transfer ediyor.
Bellek "her şeyi hafifçe iyileştiren" bir şey değil, domain'e özgü içerik öğrendi.

**Bulgu 3 — Rastgele-özellik adaptörü etkisi (H3 için alarm):**
E2 (rastgele donuk tablo!) bile python'da −0.0367 kazandı. Bu, enjeksiyon mekanizmasının
kendisinin küçük bir adaptasyon sağladığını gösterir. Gerçek python-Engram'ın (C3)
bu eşiği GEÇMESİ gerekir ki "içerik iş yapıyor" diyebilelim.

**Bulgu 4 — H1 henüz açık:** Sağlıklı (fp32-fix'li) bir Python-Engram koşusu hâlâ yok.
C2 çürük elma olduğu için "Engram vs LoRA" sorusunun cevabı C3'te gelecek.

---

## BÖLÜM 4 — ŞU AN KOŞAN TEST: C3 (Kaggle T4)

### Ne bu?

fp32-fix'li, v2 geometrili (collision 12× azaltılmış), **Python domain'inde eğitilen**
ilk sağlıklı Engram koşusu. Projeye başladığımızdaki ana hipotezin (H1/H3) gerçek testi.

### Koşu detayları

| Parametre | Değer |
|---|---|
| Platform | Kaggle, Tesla T4 (16GB), Internet ON |
| Model | Qwen/Qwen3-0.6B (donuk) |
| dtype otomatiği | T4'te bf16 destekli → bfloat16 (resolve_dtype seçti) |
| Injection | katman 1/4/7, paylaşılan tablo (~123M param, fp32) |
| Eğitim | 8000 adım × 2048 token ≈ 16M token, AdamW, tablo lr 5e-4 / diğer 1e-4 |
| Veri | data/python_1b (starcoderdata python, 1B token havuzundan) |
| Süre tahmini | prep ~1-1.5sa + train ~2-3sa + eval ~15dk ≈ toplam 4-5sa |

### Zincir adımları (kernel script otomatik yapar)

```
prep  → starcoderdata-python + fineweb paketleme (her biri 1B token)
train → C3 eğitimi (checkpoint her 2000 adımda)
eval  → üç ölçüm:
   (1) eval_base_cloud.txt        : bulut T4'teki base referansı
   (2) eval_C3_python_block.txt   : C3 vs base (python sabit bloklar)
   (3) eval_C3_general_block.txt  : C3'ün genel metne etkisi (H2 kontrolü)
```

### C3 bittiğinde karar anahtarı

| Gözlem | Anlamı | Sonraki adım |
|---|---|---|
| ΔC3 < −0.04 ve \|ΔC3\| > \|ΔE2=−0.037\| | İçerik öğreniyor + mekanizma güçlü (H3 ✓) | Uzun koşu (50M+ token), hiperparametre taraması, belki daha büyük model |
| ΔC3 ∈ [−0.037, 0) | Mekanizma çalışıyor ama python içeriği ekstra katkı vermiyor | Domain değiştir (doygun python yerine base'in zayıf olduğu alan) |
| ΔC3 ≥ 0 (E2'den kötü) | Eğitilen bellek zararlı oluyor — ciddi mimari soru işareti | Injection stratejisi yeniden düşünülür; negatif sonuç olarak da belgelenir |
| Her durumda | ΔC3py(genel blokta) ≈ 0 ise | H2 ✓ python belleği genel metni bozmuyor |

**⚠️ Karar tablosuna ek satır (H1 ayrı raporlanır):** H3'ü geçmek H1'i geçmek değildir.
Raporda mutlaka iki sayı ayrı ayrı verilmeli:
- `ΔC3 vs E2 eşiği (−0.037)` → içerik etkisi
- `ΔC3 vs LoRA (−0.2085)` → adaptasyon yöntemleri arası konum
Örn. ΔC3 = −0.06 olsa bile: "E2'yi geçti, mekanizma canlı" AMA "LoRA'nın %29'unda" — ikisi ayrı cümle.

### ⚠️ Platform-geçiş kontrol listesi (laptop → Kaggle)

"Sessiz başarısızlık" sınıfına karşı önlemler:

| Kontrol | Yöntem | Referans (laptop) |
|---|---|---|
| Veri determinizmi | Kaggle console'da val bölgesi hash'i al, aşağıdakiyle karşılaştır | `val-başı sha256: b427f5b911c8...` |
| Stream determinizmi | train bölgesinden örnek hash | `train-500M sha256: 464a1e90...` |
| fp32 sigortası | train script başında otomatik assert (eklendi) | "tum egitilebilir parametreler fp32 ✓" logu |
| Öğrenme canlılığı | step 500'de tablo norm drift kontrolü (eklendi) | drift ≥ 0.005 beklenir; altındaysa BÜYÜK uyarı |
| Checkpoint gerçekten yazıldı mı? | save sonrası dosya boyutu kontrolü (eklendi) | ~216 MB beklentisi |

Kaggle console'da hash doğrulama komutu:

```python
import sys, os, hashlib
os.chdir("/kaggle/working/engram-domain-memory"); sys.path.insert(0, ".")
import numpy as np
from src.data_loader import PackedTokenDataset
ds = PackedTokenDataset("/kaggle/working/data/python_1b", val_tokens=10_000_000)
for off, label in [(0, "val-basi"), (5242880, "val-orta")]:
    span = ds._read_span(ds.train_end + off, 65536).astype(np.int32)
    print(label, hashlib.sha256(span.tobytes()).hexdigest())
span_t = ds._read_span(500000000, 65536).astype(np.int32)
print("train-500M", hashlib.sha256(span_t.tobytes()).hexdigest())
```

Hash'ler laptop referanslarıyla TUTMUYORSA: C3 sonuçları laptop'taki B/E2 deltalarıyla
doğrudan kıyaslanamaz → ya shard'ları Kaggle dataset olarak taşı, ya tüm koşuların eval'ini
aynı ortamda yenile.

### Referans eşikleri (laptop ölçümlerinden, T4'te mutlak değerler kayabilir — deltalar kıyaslanacak)

- E2 rastgele-tablo eşiği: **−0.0367** (C3 bunu geçmeli)
- LoRA üst referansı: **−0.2085**
- D'nin kendi-domain kazanımı (güzel örnek): **−0.1132**

---

## BÖLÜM 5 — GELECEK PLAN (öncelik sırasıyla)

1. **C3 sonuçlarını yorumla** (bu dokümanın Bölüm 4 anahtarıyla)
2. Sonuca göre: uzun koşu / domain değişimi / mimari revizyon
3. Task-level eval ekle (HumanEval/MBPP) — perplexity tek başına ikna edici değil
4. Faz 3 demosu: aynı Engram tablosunu iki farklı modele takmak (LoRA'nın yapamadığı şey)
5. HF token rotate (chat + kernel içinde ifşa edilen eski token)
