# RW-VAE v2.1 pipeline, LOCAL ENGINEERING TEST on Imagenette (Windows, RTX 4060 8 GB, one process).
# Exercises the real production path (real f16d32 tokenizer, DINOv2-L, LightningDiT-B/1, pytorch-fid, LPIPS) at
# tiny budgets (configs/local_imagenette.yaml). Its numbers are NOT thesis results and never go into
# ImageNet-1k tables. Torchrun is not available on Windows (torch 2.12 without libuv), so 06 runs in one process.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_local_imagenette.ps1 [-From 03]
#
# -From STAGE skips the stages before STAGE (00 01 02 03 04 05 05b 06 07). A finished stage refuses to re-run
# into its run directory (pass --overwrite by hand). Outputs: runs\local_imagenette\<stage>\...;
# report: runs\local_imagenette\report\REPORT.md. Assets: E:\GraduateDesign\assets (see README).
param([string]$From = "00")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$Py = if ($env:PY) { $env:PY } else { ".venv\Scripts\python.exe" }
$C = @("--config", "configs/dataset_imagenet256.yaml", "--config", "configs/baseline_common.yaml",
       "--config", "configs/arms.yaml", "--config", "configs/local_imagenette.yaml")
$Exp = "runs/local_imagenette"

function Test-Stage([string]$Id) { return [string]::CompareOrdinal($Id, $From) -ge 0 }   # 05 < 05b < 06

function Invoke-Step([string]$Title) {
    # remaining arguments: the python command line; a non-zero exit code stops the pipeline
    Write-Host ""
    Write-Host "=== [local_imagenette] $Title"
    $ErrorActionPreference = "Continue"   # python's stderr (warnings) must not abort a redirected run in PS 5.1
    & $Py @args
    $code = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    if ($code -ne 0) { throw "step failed (exit $code): $Title" }
}

if (Test-Stage "00") { Invoke-Step "00 verify the allocation math" scripts/00_verify_math.py @C --figures }
if (Test-Stage "01") { Invoke-Step "01 manifest + DATA_CARD" scripts/01_prepare_manifest.py @C }
if (Test-Stage "02") { Invoke-Step "02 latent cache (all splits, flips) + re-encode check" scripts/02_cache_latents.py @C }
if (Test-Stage "03") {
    Invoke-Step "03 joint [mu; DINO] statistics (500 train_fit images; alignment gate)" `
        scripts/03_estimate_stats.py @C --max-images 500 --figures
}
if (Test-Stage "04") { Invoke-Step "04 allocations + go/no-go" scripts/04_solve_channel.py @C --figures }
if (Test-Stage "05") {
    Invoke-Step "05 generator-free metrics on train_dev (B0, B1-B3 at R=22, w=2)" `
        scripts/05_eval_recon_probe.py @C --split train_dev --R 22 --w 2 --figures
}
if (Test-Stage "05b") {
    Invoke-Step "05b decoder adaptation (B3 R=22 w=2)" scripts/05b_adapt_decoder.py @C --arm B3 --R 22 --w 2 --val-images 32
}
if (Test-Stage "06") {
    Invoke-Step "06 DiT, B0" scripts/06_train_dit.py @C --arm B0
    Invoke-Step "06 DiT, B3 R=22 w=2 (exploratory)" scripts/06_train_dit.py @C --arm B3 --R 22 --w 2 --exploratory
}
if (Test-Stage "07") {
    $Ref = "val=$Exp/fid_ref/val_n1000.npz"
    Invoke-Step "07 FID reference (val, 1000 images)" scripts/07_evaluate.py ref-stats @C --split val --max-images 1000
    Invoke-Step "07 generate B0" scripts/07_evaluate.py generate @C --arm B0 --ref-stats $Ref --save-grid 16
    Invoke-Step "07 generate B3" scripts/07_evaluate.py generate @C --arm B3 --R 22 --w 2 --ref-stats $Ref --save-grid 16
    Invoke-Step "07 report" scripts/07_evaluate.py report @C
    Write-Host "report: $Exp/report/REPORT.md (engineering test, not thesis results)"
}
