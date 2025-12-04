<#
scripts/recommend_config.ps1

Detect basic machine capabilities (CPU cores, RAM, GPU VRAM) and
recommend which LLM config to use: small, medium, or large.

Usage:
  pwsh scripts/recommend_config.ps1
  # or in Windows PowerShell:
  .\scripts\recommend_config.ps1
#>

Write-Host "=== Personal-LLM Hardware Probe (Windows) ===" -ForegroundColor Cyan

# -------------------------------
# CPU cores
# -------------------------------
try {
    $cores = [int]$env:NUMBER_OF_PROCESSORS
} catch {
    $cpuInfo = Get-CimInstance Win32_Processor | Select-Object -First 1
    $cores = $cpuInfo.NumberOfLogicalProcessors
}
Write-Host ("CPU cores detected: {0}" -f $cores)

# -------------------------------
# RAM (in GiB)
# -------------------------------
$cs = Get-CimInstance Win32_ComputerSystem
$memBytes = [double]$cs.TotalPhysicalMemory
$memGB = [math]::Round($memBytes / 1GB)
Write-Host ("RAM detected:       {0} GiB" -f $memGB)

# -------------------------------
# GPU VRAM (in GiB)
# -------------------------------
# Take the largest GPU memory if multiple
$gpuGB = 0
try {
    $gpus = Get-CimInstance Win32_VideoController | Where-Object { $_.AdapterRAM -gt 0 }
    if ($gpus) {
        $maxAdapter = $gpus | Sort-Object -Property AdapterRAM -Descending | Select-Object -First 1
        $gpuGB = [math]::Round($maxAdapter.AdapterRAM / 1GB)
    }
} catch {
    $gpuGB = 0
}
Write-Host ("GPU VRAM detected:  {0} GiB" -f $gpuGB)

Write-Host "------------------------------------------"

# -------------------------------
# Recommendation logic
# -------------------------------
# Same heuristic as Linux:
#   - large:  GPU >= 8GB  OR (RAM >= 32GB AND CORES >= 8)
#   - medium: GPU >= 4GB  OR (RAM >= 16GB AND CORES >= 4)
#   - small:  otherwise
#
$recommendation = "small"
$configPath = "config/small.yaml"

if ( ($gpuGB -ge 8) -or ( ($memGB -ge 32) -and ($cores -ge 8) ) ) {
    $recommendation = "large"
    $configPath = "config/large.yaml"
}
elseif ( ($gpuGB -ge 4) -or ( ($memGB -ge 16) -and ($cores -ge 4) ) ) {
    $recommendation = "medium"
    $configPath = "config/medium.yaml"
}

Write-Host ("Recommended model size:  {0}" -f $recommendation) -ForegroundColor Green
Write-Host ("Recommended config file: {0}" -f $configPath)
Write-Host ""

Write-Host "Example command:" -ForegroundColor Yellow
Write-Host ("  python personal_llm.py --config {0} --text_file data/sample_corpus.txt --generate" -f $configPath)

Write-Host ""
Write-Host "Note: You can tweak the thresholds in scripts/recommend_config.ps1 if needed."