# Gece zinciri v2 — bf16 yuvarlama bug'i duzeltildikten sonra.
# E2 -> eval -> D -> FINAL_RESULTS + TAMAMLANDI isareti (faz 2'yi tetikler)

$ErrorActionPreference = "Continue"
$env:TOKENIZERS_PARALLELISM = "false"
$log = "runs\overnight\pipeline_log.txt"

function Log($msg) {
    $stamp = Get-Date -Format "HH:mm:ss"
    "$stamp  $msg" | Tee-Object -FilePath $log -Append
}
function Wait-ForProcessExit($pattern) {
    while ($true) {
        $p = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
            Where-Object { $_.CommandLine -match $pattern }
        if (-not $p) { return }
        Start-Sleep -Seconds 60
    }
}

Log "=== GECE ZINCIRI V2 BASLADI (fp32 fix) ==="

Log "E2 egitimi basliyor (tablo donuk, bsz4)..."
python train/train_engram.py --data-dir data/python_1b --steps 6000 --bsz 4 --seq-len 512 `
    --eval-interval 1500 --freeze-table --alpha-init 0.05 --table-lr 5e-4 `
    --out-dir runs/E2_random 2>&1 | Add-Content $log
Log "E2 bitti."

Log "D egitimi oncesi FineWeb bekleniyor..."
Wait-ForProcessExit "general_1b"
Log "FineWeb hazir. D egitimi basliyor (bsz3, fp32 tablo icin guvenli boyut)..."
python train/train_engram.py --data-dir data/general_1b --steps 8000 --bsz 3 --seq-len 512 `
    --eval-interval 2000 --alpha-init 0.05 --table-lr 5e-4 `
    --out-dir runs/D_general 2>&1 | Add-Content $log
Log "D bitti."

Log "FINAL EVAL..."
python eval/eval_fixed.py --lora runs/B_lora/adapter_step2000 `
    --engrams @("C2=runs/C2_python/engram_step6000.pt", "E2=runs/E2_random/engram_step6000.pt", "D=runs/D_general/engram_step8000.pt") `
    --out runs\overnight\FINAL_RESULTS.txt 2>&1 | Add-Content $log

"$(Get-Date -Format 'HH:mm:ss')  === GECE ZINCIRI TAMAMLANDI ===" | Add-Content $log
