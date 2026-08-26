# Gece otomasyon zinciri:
#   C-v2 bitisini bekle -> eval -> E2 egitimi -> eval -> FineWeb bitisini bekle
#   -> D egitimi -> eval -> FINAL raporu
# Her adim runs/overnight/ altina loglanir.

$ErrorActionPreference = "Continue"
$env:TOKENIZERS_PARALLELISM = "false"
$log = "runs\overnight\pipeline_log.txt"
New-Item -ItemType Directory -Force -Path "runs\overnight" | Out-Null

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

function Run-Eval($engrams, $outFile, $label) {
    Log "EVAL basliyor: $label"
    python eval/eval_fixed.py --lora runs/B_lora/adapter_step2000 --engrams @($engrams) --out $outFile 2>&1 |
        ForEach-Object { "$_" } | Add-Content $log
    Log "EVAL bitti: $label -> $outFile"
}

Log "=== GECE ZINCIRI BASLADI ==="

# --- Adim 1: C-v2 bitmesini bekle ---
Log "C-v2 bitisi bekleniyor..."
Wait-ForProcessExit "C2_python"
Log "C-v2 bitti."

# --- Adim 2: C-v2 eval ---
Run-Eval @("C2=runs/C2_python/engram_step6000.pt") "runs\overnight\eval_C2.txt" "C2"

# --- Adim 3: E2 egitimi (rastgele donuk tablo + ogrenen reader) ---
Log "E2 egitimi basliyor..."
python train/train_engram.py --data-dir data/python_1b --steps 6000 --bsz 4 --seq-len 512 `
    --eval-interval 1000 --freeze-table --alpha-init 0.05 --table-lr 5e-4 `
    --out-dir runs/E2_random 2>&1 | Add-Content $log
Log "E2 egitimi bitti."

Run-Eval @("E2=runs/E2_random/engram_step6000.pt") "runs\overnight\eval_E2.txt" "E2"

# --- Adim 4: FineWeb paketinin bitmesini bekle ---
Log "FineWeb paketi bekleniyor..."
Wait-ForProcessExit "general_1b"
Log "FineWeb hazir."

# --- Adim 5: D egitimi (general Engram) ---
Log "D egitimi basliyor..."
python train/train_engram.py --data-dir data/general_1b --steps 6000 --bsz 4 --seq-len 512 `
    --eval-interval 1000 --alpha-init 0.05 --table-lr 5e-4 `
    --out-dir runs/D_general 2>&1 | Add-Content $log
Log "D egitimi bitti."

Run-Eval @("D=runs/D_general/engram_step6000.pt") "runs\overnight\eval_D.txt" "D"

# --- Adim 6: FINAL birlesik rapor ---
Run-Eval @(
    "C2=runs/C2_python/engram_step6000.pt",
    "E2=runs/E2_random/engram_step6000.pt",
    "D=runs/D_general/engram_step6000.pt"
) "runs\overnight\FINAL_RESULTS.txt" "FINAL (tum kosular)"

Log "=== GECE ZINCIRI TAMAMLANDI ==="
