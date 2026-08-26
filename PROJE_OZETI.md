# Domain-Spesifik Engram Projesi — Ne, Neden, Nasıl?

> **Tek cümlelik özet:** Donuk (değiştirilemez) bir hazır LLM'e, belirli bir domain'e (Python kodu)
> özel "harici hafıza" modülü sonradan takılıp takılamayacağını ve bunun klasik fine-tuning'den
> (LoRA) daha iyi olup olmadığını test ediyoruz.

---

## 1. Amaç Neyi Araştırıyoruz?

### Problem
Büyük dil modelleri bilgiyi ağırlıklarında ("parametrik") saklar. Bir modele yeni bir alan
(örn. Python programlama) öğretmek istersen klasik yöntem **fine-tuning**'dir: model ağırlıkları
üzerinde eğitim yaparsın.

### Fikir
DeepSeek'in 2026 başında duyurduğu **Engram** mimarisinden esinlendik: modelin dışında,
n-gram hash tabanlı O(1) erişilebilir bir "statik bellek" modülü. DeepSeek bu belleği modeli
*eğitirken* sıfırdan birlikte eğitti — yani normal bir hazır LLM'e sonradan eklenemiyordu.

**Bizim sorumuz:** Hiç Engram'sız eğitilmiş, tamamen donuk bir modele (Qwen3-0.6B),
sonradan Python'a özel bir Engram belleği *ekleyip* sadece belleği mi eğitsek,
model Python'da gelişir mi?

### Test edilecek hipotezler

| # | Hipotez | Anlamı |
|---|---|---|
| H1 | Aynı veriyle **Engram > LoRA** | Ana başarı kriteri; LoRA'yı geçemezse fikir değersiz |
| H2 | ΔPython > ΔGeneral | Kazanç domain'e özgü olmalı; genel yetenekler zarar görmemeli |
| H3 | Rastgele içerikli bellek kazanım vermez | Etki içerikten gelir, sadece ekstra kapasiteden değil |
| H4 | Belleği çıkarınca model birebir orijinale döner | "Tak-çıkar" güvenliği |

### Neden önemli / farkı ne?
- LoRA adapter'lar **modele kilitlidir** (ağırlık boyutlarına dikilir). Engram tablosu metne
  endeksli → teorik olarak **modeller arası taşınabilir**.
- Tek donuk base + N değiştirilebilir domain belleği = modüler dağıtım hayali.
- Literatürde bu senaryo (Engram'sız modele sonradan domain belleği ekleme) test edilmemiş boşluk.

---

## 2. Ne Kullanıldı?

### Donanım & yazılım

| Bileşen | Seçim | Not |
|---|---|---|
| GPU | NVIDIA RTX 3070 Ti Laptop (8 GB) | Windows/WDDM ortamı, paylaşımlı bellek tuzağına dikkat |
| PyTorch | 2.6.0 + cu126 | Başta CPU-only kurulmuştu; CUDA sürümüne geçildi |
| Transformers | 5.15.1 | Qwen3 desteği için 4.48 → 5.15 yükseltildi |
| peft | 0.20.0 | LoRA baseline (B koşusu) için |
| Veriseti kütüphanesi | datasets 3.2.0 | Streaming modunda |

### Model

| Model | Rol |
|---|---|
| **Qwen/Qwen3-0.6B** (donuk) | Pilot base model. Küçük seçildi ki Python headroom olsun ve hızlı iterasyon yapılsın. Coder-model bilerek seçilmedi. |
| Qwen3.5-0.8B | Faz 3'te teyit planlanan ikinci model (multimodal + hibrit attention olduğu için pilotda değil) |

### Verisetleri

| Veriseti | Amaç | Nasıl kullanıldı |
|---|---|---|
| `bigcode/starcoderdata` (python alt kümesi, gated) | Eğitim + validasyon | **Streaming**: veriseti diske inmeden akıtılıdı → filtre → tokenize → paketlenmiş `.bin` shard'lar. Sonuç: **1,003 milyar token (~4 GB), 19 dakika** |
| `HuggingFaceFW/fineweb` | D koşusu (general kontrol) | Aynı pipeline, henüz paketlenmedi |

Veri güvenliği: StarCoderData benchmark'lara karşı zaten dekontamine edilmişti
(HumanEval/MBPP sızıntısı riski düşük); spot-check yapılacak.

### Referans makaleler

1. **Engram** (arXiv:2601.07372, DeepSeek) — ana mimari ilham; hash şeması demo kodundan uyarlandı
2. **Tokenizer-Agnostic Engram** (arXiv:2607.29065) — Faz 3 tokenizer bağımsızlığı için
3. **Memory Grafting** (arXiv:2605.20948) — offline bellek inşası alternatifi
4. **Cross-Model Memory Transfer** (arXiv:2608.17050) — reader mimarisi referansı

---

## 3. Nasıl Kullanıldı? (Mimari)

### Engram enjeksiyonu nasıl çalışıyor?

```
token dizisi ──► n-gram hash (2-gram + 3-gram, XOR şema, katman-bazlı multiplier)
                       │
                       ▼
              paylaşılan embedding tablosu (110M param, ~300K satır × 8 head)
                       │
                       ▼
        value/key projeksiyonu + içerik-bazlı sigmoid gate + short conv
                       │
                       ▼
   h_yeni = h_eski + α · engram_çıktısı        ← α zero-init (kritik!)
```

- **Takılma noktası:** Katman 1, 4 ve 7'nin girişi (erken katmanlar — orijinal makalenin
  "erken katmanlar statik işi belleğe devreder" bulgusuna uygun), HF hook'larıyla
- **α (alpha) gate zero-init:** Eğitimin ilk anında modül matematiksel olarak yok gibidir →
  model birebir korunur. Bu, H4'ün (tak-çıkar) teknik garantisi
- **Paylaşılan tablo:** 3 injection noktası TEK embedding tablosunu paylaşır → bütçe ~333M değil **113M**
- **Eğitilebilir:** sadece tablo + projeksiyonlar + gate'ler (113M). Backbone'un tamamı donuk

### Veri pipeline'ı

```
HF stream (veri inmeyerek okunur)
  → dosya filtreleri (boyut, satır uzunluğu, alfanumerik oranı)
  → Qwen tokenizer ile paralel tokenize (16MB buffer, rayon parallelism)
  → kesintisiz int32 token akışı → 128M'lik .bin shard'lar
  → eğitim: memmap + rastgele crop
```

Disk dostudur: 25 GB boş alanda bile çalışır (tam veriseti asla indirilmez).

---

## 4. Deney Matrisi

| Koşu | Trainable | Amaç | Durum |
|---|---|---|---|
| A — Base | 0 | Referans | Bekliyor |
| B — LoRA | ~113M (r kalibre) | Adaptasyon baseline'ı (H1 rakibi) | Script hazır |
| **C — Python Engram** | 113M | **Asıl hipotez** | **Eğitiliyor** (step 1000/2000) |
| D — General Engram | ~113M | Domain specificity kontrolü (H2) | Veri pipeline'ı hazır |
| E2 — Random Engram | 113M | İçerik vs mekanizma (H3) | Script hazır |
| F — QLoRA/full | bütçeye göre | Üst sınır | Sonra |
| G — RAG | 0 | Parametrik olmayan alternatif | Sonra |

Karşılaştırma metni: held-out **val loss/ppl** (`eval_ppl.py`) + ileride HumanEval/MBPP.

---

## 5. Şimdiye Kadar Yapılanlar (ve Bulunan Buglar)

### Kurulum aşaması
1. CUDA'lı torch kurulumu (CPU-only sürüm hata maskelemişti)
2. transformers 5.x yükseltmesi, gated HF dataset erişimi (token + lisans)
3. Streaming veri pipeline'ı: 1B token'lık Python paketi 19 dk'da üretildi

### Yazılan çekirdek kod (`src/engram/`)
- `hashing.py` — n-gram hash + compressed tokenizer (+ GPU-side hash)
- `module.py` — paylaşılan multi-head embedding + gated injection (zero-init α)
- `wrapper.py` — HF modele hook ile takma/çıkarma/dondurma
- `train/train_engram.py`, `train/train_lora.py`, `eval/eval_ppl.py`, `eval/report.py`

### Doğrulanan garantiler (testlerle kanıtlandı)
- ✅ α=0 iken logits **bit-bit aynı** (tak-çıkar güvenliği)
- ✅ Sökünce (remove) model birebir orijinal
- ✅ Gradyan sadece Engram parametrelerine akıyor; backbone donuk
- ✅ Paylaşılan tablo optimizer'da tek kez sayılıyor

### Yakalanan buglar (hepsi düzeltildi)

| Bug | Belirti | Kök neden |
|---|---|---|
| Her katmana farklı hash asalları | device-side assert (CUDA) | Tablo katman 1 boyutuylayla sınırlıydı, katman 4/7 taşmış |
| Paylaşılan tablo 3× optimizer'da | Etkili LR 3 katı | `parameters()` kimlik dedupe yapmıyordu |
| CPU-side hashing | %35 yavaşlık | numpy hesap her adımda CPU'daydı |
| Çift label-shift | Loss 13.0 (gerçek: 1.67) | `labels=y` verilince transformers zaten kaydırıyor; bir kez daha kaydırılmıştı |
| WDDM shared-memory tuzağı | 400 tok/s duvarı | 16 GB kullanım = sistem RAM'ine taşma; konfig küçültülünce saf VRAM'de 2030 tok/s |

### Performans seyri
```
400 tok/s (RAM spill) → 660 (GPU hash) → 2030 tok/s (doğru konfig: bsz4×512, 5.4 GB)
```

---

## 6. Şu Anki Durum

- **C koşusu eğitiliyor**: step 1000/2000 (~%50), tahmini bitiş 20:47
- İlk sinyaller umut verici: val ppl 4.54'e indi, gate'ler (α) sıfırdan ayrışmaya başladı
- Sırada: A referans ölçümü → B (LoRA) koşusu → karşılaştırma raporu

## 7. Açık Konular

- [ ] HumanEval/MBPP entegrasyonu (task-level değerlendirme)
- [ ] D koşusu için FineWeb paketi
- [ ] E2 random-memory koşusu
- [ ] Faz 3: çapraz-model taşıma + çoklu bellek routing demosu
- [ ] HF token rotate (chatta paylaşıldığı için işler bitince mutlaka yenilenmeli)
