# C3: fp32 fix SONRASI python Engram — gercek H1/H3 testi
# 1) C3 egitimi (8000 adim)
# 2) Paired eval: A, B, E2, D, C3 (python bloku)
# 3) Capraz-domain: C3'u genel blokta olc
# 4) Guncel sabah raporu

$ErrorActionPreference = "Continue"
Set-Location "C:\Users\ilker\deneme"
$env:TOKENIZERS_PARALLELISM = "false"
$log = "runs\overnight\c3_log.txt"

function Log($msg) {
    $stamp = Get-Date -Format "HH:mm:ss"
    "$stamp  $msg" | Tee-Object -FilePath $log -Append
}

Log "=== C3 KOSUSU BASLADI ==="
python train/train_engram.py --data-dir data/python_1b --steps 8000 --bsz 4 --seq-len 512 `
    --eval-interval 2000 --alpha-init 0.05 --table-lr 5e-4 `
    --out-dir runs/C3_python 2>&1 | Add-Content $log
Log "C3 egitimi bitti."

Log "Paired eval (python bloku)..."
python eval/eval_fixed.py --lora runs/B_lora/adapter_step2000 `
    --engrams @("E2=runs/E2_random/engram_step6000.pt", "D=runs/D_general/engram_step8000.pt", "C3=runs/C3_python/engram_step8000.pt") `
    --out runs\overnight\eval_C3_python_block.txt 2>&1 | Add-Content $log

Log "Capraz-domain eval (genel blokta C3)..."
python eval/eval_fixed.py --data-dir data/general_1b `
    --lora none `
    --engrams @("C3py=runs/C3_python/engram_step8000.pt") `
    --out runs\overnight\eval_C3_general_block.txt 2>&1 | Add-Content $log

Log "=== C3 ZINCIRI TAMAMLANDI ==="
