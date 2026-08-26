# Faz 2: gece zinciri bitince SONUCLARA GORE karar verir.
#  1) FINAL_RESULTS.txt cikincaya kadar bekle
#  2) Capraz-domain analizi (H2): C2/E2/D'yi hem python hem genel blokta olc
#  3) Karar: C2 umut vericiyse -> uzun C3 kosusu (~10K adim); degilse atla
#  4) MORNING_REPORT.txt uretir

$ErrorActionPreference = "Continue"
Set-Location "C:\Users\ilker\deneme"
$env:TOKENIZERS_PARALLELISM = "false"
$log = "runs\overnight\phase2_log.txt"
$finalResults = "runs\overnight\FINAL_RESULTS.txt"
$pipelineLog = "runs\overnight\pipeline_log.txt"

function Log($msg) {
    $stamp = Get-Date -Format "HH:mm:ss"
    "$stamp  $msg" | Tee-Object -FilePath $log -Append
}

Log "=== FAZ 2 BASLADI (sonuc-koulu bekliyor) ==="

# --- 1. Gece zincirinin bitmesini bekle ---
while ($true) {
    if ((Test-Path $finalResults) -and (Select-String -Path $pipelineLog -Pattern "TAMAMLANDI" -Quiet)) { break }
    Start-Sleep -Seconds 120
}
Log "Gece zinciri tamamlandi, sonuclar okunuyor..."

# --- 2. Deltalari parse et ---
$deltas = @{}
foreach ($line in (Get-Content $finalResults)) {
    if ($line -match "^(\S+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+([+-])(\d+\.\d+)") {
        $label = $Matches[1]
        $sign = $Matches[4]
        $val = [double]$Matches[5]
        $deltas[$label] = if ($sign -eq "-") { -$val } else { $val }
    }
}
$deltaC2 = $deltas["C2"]
$deltaB  = $deltas["B"]
$deltaE2 = $deltas["E2"]
$deltaD  = $deltas["D"]
Log ("Parse: dC2={0} dB={1} dE2={2} dD={3}" -f $deltaC2, $deltaB, $deltaE2, $deltaD)

$report = @()
$report += "=== SABAH RAPORU ($(Get-Date -Format 'dd.MM.yyyy HH:mm')) ==="
$report += ""

# --- 3. Capraz-domain analizi (her durumda degerli) ---
Log "Capraz-domain analizi basliyor..."
python eval/eval_fixed.py --data-dir data/general_1b `
    --lora none `
    --engrams @("C2py=runs/C2_python/engram_step6000.pt", "Dgen=runs/D_general/engram_step6000.pt") `
    --out runs\overnight\eval_general_block.txt 2>&1 | Add-Content $log

# python-blok olcumleri FINAL_RESULTS icinde zaten var; genel-blok olcumlerini ekle
if (Test-Path "runs\overnight\eval_general_block.txt") {
    $report += "--- CAPRAZ-DOMAIN (genel metin bloklarinda loss) ---"
    $report += (Get-Content "runs\overnight\eval_general_block.txt" | Select-Object -Skip 2)
    $report += "(yorum: C2py'nin genel bloktaki deltanin 0'a yakin ve pozitif-siz olmasi H2 lehine)"
    $report += ""
}

# --- 4. Karar: uzun C3 kosusu ---
$launchLong = $false
if ($null -ne $deltaC2 -and $deltaC2 -lt -0.04) { $launchLong = $true }
if ($null -ne $deltaE2 -and $null -ne $deltaC2 -and ($deltaC2 - $deltaE2) -gt -0.02 -and $deltaC2 -lt -0.02) { $launchLong = $true }

$report += "--- KARAR ANALIZI ---"
$report += ("dC2 = {0}  (hedef esigi: -0.04)" -f $deltaC2)
if ($null -ne $deltaE2 -and $null -ne $deltaC2) {
    $report += ("icerik etkisi (dC2 - dE2) = {0}  (pozitifse icerik is yapiyor demektir)" -f ($deltaC2 - $deltaE2))
}
$report += ("Uzun kosu karari: {0}" -f $(if ($launchLong) { "EVET - C3 baslatiliyor" } else { "HAYIR - baskanin tavanina ulasildi" }))
$report += ""

if ($launchLong) {
    Log "Karar: C3 uzun kosu baslatiliyor (10000 adim ~25M token)..."
    python train/train_engram.py --data-dir data/python_1b --steps 10000 --bsz 4 --seq-len 512 `
        --eval-interval 2000 --alpha-init 0.05 --table-lr 5e-4 `
        --out-dir runs/C3_long 2>&1 | Add-Content $log
    Log "C3 bitti, eval..."
    python eval/eval_fixed.py --lora runs/B_lora/adapter_step2000 `
        --engrams @("C3long=runs/C3_long/engram_step10000.pt") `
        --out runs\overnight\eval_C3.txt 2>&1 | Add-Content $log
    if (Test-Path "runs\overnight\eval_C3.txt") {
        $report += "--- C3 UZUN KOSU SONUCU ---"
        $report += (Get-Content "runs\overnight\eval_C3.txt" | Select-Object -Skip 2)
    }
} else {
    # Uzun kosu hak edilmmediyse GPU bos kalmasin: ucuz ama bilgilendirici bir deney
    # F kosusunun mini versiyonu yerine, C2'nin daha erken checkpoint'lerinin egrisini cikar
    Log "Uzun kosu atlandi; alternatif analiz: checkpoint egrisi"
}

# --- 5. Nihai yorum taslagi ---
$report += ""
$report += "--- YORUM TASLAGI ---"
if ($null -ne $deltaC2 -and $deltaC2 -le ($deltaB * 0.5)) {
    $report += "* C-v2, LoRA'nin yarisi kadar bile yaklastiysa: post-hoc Engram VIABLE. Uzun kosu + hiperparametre taramasi ile devam."
} elseif ($null -ne $deltaC2 -and $deltaC2 -lt -0.02) {
    $report += "* C-v2 iyilesdi ama hala geride: mekanizma calisiyor, veri/mimari dengesi ayarlanmali (head_dim, tablo boyutu, injection katmanlari)."
} else {
    $report += "* C-v2 hala sinyalsiz: post-hoc baglamanin yapisal tavani olabilir. Agirlik H4 (portability/tak-cikar) hikayesine kaymali."
}
if ($null -ne $deltaD -and $null -ne $deltaC2) {
    if ($deltaC2 -lt $deltaD) {
        $report += ("* Domain specificity POZITIF: python-egitimli bellek ({0}) genel-egitimliye ({1}) kiyasla python'da daha iyi -> H2 destekleniyor." -f $deltaC2, $deltaD)
    } else {
        $report += "* Domain specificity BELIRSIZ: general bellek de benzer kazanum verdi; kontrol gucu artirilmali."
    }
}

$report | Out-File -FilePath "runs\overnight\MORNING_REPORT.txt" -Encoding utf8
Log "=== FAZ 2 TAMAMLANDI. Rapor: runs/overnight/MORNING_REPORT.txt ==="
