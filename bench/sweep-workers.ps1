param(
    [Parameter(Mandatory)][string]$SourceSha,
    [string]$Source = 'models/phase9-workers',
    [string]$OutDir = 'bench/results/phase9-workers-2026-11-17',
    [string]$Racket = 'C:\Program Files\Racket\Racket.exe'
)

# Screening only. Final validation alternates the selected tree and baseline.
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$sourcePath = Join-Path $root $Source
$dest = Join-Path $root $OutDir
if (Test-Path $dest) { throw 'Use a fresh output directory' }
New-Item -ItemType Directory -Path $dest | Out-Null
$rows = [Collections.Generic.List[object]]::new()
foreach ($job in @(@('decode', 128), @('prefill', 128), @('head', 1))) {
    foreach ($workers in @(1, 2, 4, 6, 12 | Where-Object { $_ -le [Environment]::ProcessorCount })) {
        Write-Host "$($job[0]) $($job[1]), workers $workers"
        $flag = if ($job[0] -eq 'prefill') { '--prefill-workers' } else { '--decode-workers' }
        $text = & $Racket (Join-Path $root 'bench/compare-phases.rkt') --source $sourcePath `
            --model-dir (Join-Path $root 'models/gpt2') --child $job[0] --size $job[1] --runs 3 $flag $workers
        if ($LASTEXITCODE -ne 0) { throw 'Worker trial failed' }
        $data = ($text -join "`n") | ConvertFrom-Json
        $row = [pscustomobject]@{kind=$job[0]; size=$job[1]; workers=$workers; data=$data}
        $rows.Add($row)
        $row | ConvertTo-Json -Depth 10 -Compress | Add-Content (Join-Path $dest 'samples.jsonl') -Encoding utf8
        Write-Host ('{0:N3} ms' -f $data.median_ms)
    }
}
$report = [pscustomobject]@{source_sha=$SourceSha; date=(Get-Date -Format 'yyyy-MM-dd');
    cpu=$env:PROCESSOR_IDENTIFIER; processors=[Environment]::ProcessorCount; results=@($rows)}
$report | ConvertTo-Json -Depth 12 | Set-Content (Join-Path $dest 'summary.json') -Encoding utf8
