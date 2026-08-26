# Devir zinciri: D -> eval -> E2 -> eval -> FINAL + TAMAMLANDI
# (E2'nin yarim kalan kosusu atilir, sifirdan egitilir)

$ErrorActionPreference = "Continue"
Set-Location "C:\Users\ilker\deneme"
$env:TOKENIZERS_PARALLELISM = "false"
$log = "runs\overnight\pipeline_log.txt"

function Log($msg) {
    $stamp = Get-Date -Format "HH:mm:ss"
    "$stamp  $msg" | Tee-Object -FilePath $log -Append
}

Log "=== ZINCIR DEVIR (v2-fix, dayanikli) BASLADI ==="

# --- 1. D egitimi (genel Engram) ---
Log "D egitimi basliyor (bsz3, fp32 tablo)..."
python train/train_engram.py --data-dir data/general_1b --steps 8000 --bsz 3 --seq-len 512 `
    --eval-interval 2000 --alpha-init 0.05 --table-lr 5e-4 `
    --out-dir runs/D_general 2>&1 | Add-Content $log
Log "D egitimi bitti."

# --- 2. E2 egitimi (rastgele donuk tablo) ---
Log "E2 egitimi basliyor..."
Remove-Item -Recurse -Force "runs\E2_random" -ErrorAction SilentlyContinue
python train/train_engram.py --data-dir data/python_1b --steps 6000 --bsz 4 --seq-len 512 `
    --eval-interval 1500 --freeze-table --alpha-init 0.05 --table-lr 5e-4 `
    --out-dir runs/E2_random 2>&1 | Add-Content $log
Log "E2 egitimi bitti."

# --- 3. Final eval ---
Log "FINAL EVAL basliyor..."
python eval/eval_fixed.py --lora runs/B_lora/adapter_step2000 `
    --engrams @("C2=runs/C2_python/engram_step6000.pt", "E2=runs/E2_random/engram_step6000.pt", "D=runs/D_general/engram_step8000.pt") `
    --out runs\overnight\FINAL_RESULTS.txt 2>&1 | Add-Content $log
Log "FINAL EVAL bitti."

"$(Get-Date -Format 'HH:mm:ss')  === GECE ZINCIRI TAMAMLANDI ===" | Add-Content $log
