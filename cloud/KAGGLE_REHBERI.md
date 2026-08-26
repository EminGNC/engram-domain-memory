# Kaggle'a Taşıma Rehberi (C3 koşusu)

## 0. Ön hazırlık (laptopunda, 10 dk)

1. Projeyi GitHub private repo'ya push et (kod + `cloud/` klasörü; `data/`, `runs/` hariç — büyük).
   Alternatif: proje zip'i yapıp Kaggle'a "Dataset" olarak yükle.
2. HF token'ını hazır tut (starcoderdata gated).

## 1. Kaggle notebook oluştur

- kaggle.com → Create → New Notebook
- Settings → Accelerator: **GPU T4 x2** (ya da P100)
- Settings → Internet: **ON**
- Add Input → GitHub repo'nuzu ekle (veya zip dataset'i)

## 2. İlk hücre — ortam

```python
import os
os.environ["HF_TOKEN"] = "hf_..."  # ya da Kaggle Secrets kullan (tercih edilir):
# from kaggle_secrets import UserSecretsClient; os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

!pip install -q -r requirements.txt
%cd /kaggle/working
!cp -r /kaggle/input/<repo-adin>/src .
!cp -r /kaggle/input/<repo-adin>/scripts .
!cp -r /kaggle/input/<repo-adin>/configs .
!cp -r /kaggle/input/<repo-adin>/train .
!cp -r /kaggle/input/<repo-adin>/eval .
```

Not: `train/` klasör adı Kaggle input'ta sorun çıkarsa `trainer/` olarak yeniden adlandır.

## 3. Fazları çalıştır (ayrı hücreler, sırayla)

```python
!python cloud/run_c3.py prep    # ~1-2 saat (T4 ağ hızına göre)
```
```python
!python cloud/run_c3.py train   # ~2-3 saat T4'te
```
```python
!python cloud/run_c3.py eval    # ~15 dk
```

## 4. Ya da tek seferde (tarayıcıyı kapatabilirsin)

Notebook'u **Save Version → Save & Run All (Commit)** ile çalıştır.
9-12 saate kadar arka planda sürer, sonuçlar output'a kaydeder.

## 5. Sonuçları alma

Notebook Output bölümünden indir:
- `runs/overnight/eval_C3_python_block.txt`  ← asıl tablo
- `runs/overnight/eval_C3_general_block.txt` ← çapraz-domain
- `/kaggle/working/runs/C3_python/engram_step8000.pt` (~500MB, istersen)

## Yorumlama anahtarı

| Karşılaştırma | Soru |
|---|---|
| ΔC3 vs ΔE2(laptop: −0.037) | C3 rastgele-tablo kontrolünü geçiyor mu? (H3) |
| ΔC3 vs ΔB(laptop: −0.209) | LoRA'ya ne kadar yaklaştı? (H1) |
| Genel blokta ΔC3py ≈ 0 mı? | Python belleği genel metne zarar vermiyor mu? (H2) |

⚠️ Dikkat: bulut GPU'su (T4) laptop GPU'sundan farklı → mutlak loss değerleri biraz değişebilir;
sadece AYNI oturumda ölçülen değerler birbiriyle karşılaştırılmalı. Base'i bulutta da ölç
(eval_all bunu yapıyor) ve deltaları ona göre hesapla.
