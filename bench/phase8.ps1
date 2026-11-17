param(
    [string]$Racket = 'C:\Program Files\Racket\Racket.exe',
    [string]$OutDir = 'bench/results/phase8-windows-2026-11-17',
    [int]$Rounds = 3,
    [string]$Phase = 'phase8',
    [string]$BaselineRef = '1ad7035',
    [string]$BaselineSource = 'models/phase8-baseline'
)

# Alternate the trees. Each sample is a fresh process with one warmup.
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$baseline = Join-Path $root $BaselineSource
if (!(Test-Path (Join-Path $baseline 'src/model.rkt'))) { throw 'Missing baseline snapshot' }
if ($Rounds -lt 3 -or $Rounds % 2 -eq 0) { throw 'Use an odd round count of at least three' }
$dest = [IO.Path]::GetFullPath((Join-Path $root $OutDir))
if (Test-Path $dest) { throw 'Use a new output directory' }
New-Item -ItemType Directory -Path $dest | Out-Null
$raw = Join-Path $dest 'samples.jsonl'
$sha = (git -C $root rev-parse HEAD).Trim()
$baselineSha = (git -C $root rev-parse $BaselineRef).Trim()
$runtime = (& $Racket --version) -join ' '
$jobs = @(@('load', 0), @('prefill', 16), @('prefill', 128), @('prefill', 512),
          @('prefill', 1000), @('decode', 128), @('sample', 128), @('head', 1), @('square', 768))
$rows = [Collections.Generic.List[object]]::new()

foreach ($job in $jobs) {
    for ($round = 1; $round -le $Rounds; $round++) {
        $order = if ($round % 2) { @('baseline', $Phase) } else { @($Phase, 'baseline') }
        foreach ($phase in $order) {
            $source = if ($phase -eq 'baseline') { $baseline } else { $root }
            Write-Host "$phase $($job[0]) $($job[1]), round $round"
            $text = & $Racket (Join-Path $root 'bench/compare-phases.rkt') --source $source `
                --model-dir (Join-Path $root 'models/gpt2') --child $job[0] --size $job[1] --runs 1
            if ($LASTEXITCODE -ne 0) { throw 'Benchmark child failed' }
            $data = ($text -join "`n") | ConvertFrom-Json
            $row = [pscustomobject]@{ phase = $phase; kind = $job[0]; size = $job[1]; round = $round; data = $data }
            $rows.Add($row)
            $row | ConvertTo-Json -Depth 10 -Compress | Add-Content -LiteralPath $raw -Encoding utf8
            Write-Host ('{0:N3} ms' -f $data.median_ms)
        }
    }
}

function Median($values) {
    $sorted = @($values | Sort-Object)
    $sorted[[int][math]::Floor($sorted.Count / 2)]
}

$summary = foreach ($job in $jobs) {
    $a = @($rows | Where-Object { $_.kind -eq $job[0] -and $_.size -eq $job[1] -and $_.phase -eq 'baseline' })
    $b = @($rows | Where-Object { $_.kind -eq $job[0] -and $_.size -eq $job[1] -and $_.phase -eq $Phase })
    $ids = $a[0].data.ids -join ','
    foreach ($row in @($a) + @($b)) {
        if (($row.data.ids -join ',') -ne $ids) { throw 'Token IDs changed between samples or trees' }
    }
    $ams = Median $a.data.median_ms
    $bms = Median $b.data.median_ms
    [pscustomobject]@{
        kind = $job[0]; size = $job[1]; baseline_ms = $ams; "${Phase}_ms" = $bms; speedup = $ams / $bms
        baseline_allocated_bytes = Median $a.data.samples_allocated_bytes
        "${Phase}_allocated_bytes" = Median $b.data.samples_allocated_bytes
        baseline_peak_bytes = Median $a.data.peak_bytes; "${Phase}_peak_bytes" = Median $b.data.peak_bytes
    }
}
$report = [pscustomobject]@{
    phase = $Phase; baseline_sha = $baselineSha; candidate_sha = $sha; racket = $runtime
    date = '2026-11-17'; cpu = 'AMD Ryzen 5 8645HS'; os = 'windows'; processors = [Environment]::ProcessorCount
    rounds = $Rounds; warmup = 1; results = @($summary)
}
$report | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $dest 'summary.json') -Encoding utf8
$lines = @("# $Phase speed comparison", '', "Candidate: ``$sha``. Baseline: ``$BaselineRef``.", '',
    "Alternating trees, one warmup per child, median of $Rounds fresh-process samples.",
    'Generation includes prompt prefill. Allocated bytes count only the timed call.',
    'Process peaks include loading and are not OS working-set measurements.', '',
    "| Configuration | Baseline ms | $Phase ms | Speedup |", '| --- | ---: | ---: | ---: |')
foreach ($row in $summary) {
    $lines += '| {0} {1} | {2:F3} | {3:F3} | {4:F2} |' -f $row.kind, $row.size, $row.baseline_ms, $row."${Phase}_ms", $row.speedup
}
$lines | Set-Content (Join-Path $dest 'comparison.md') -Encoding utf8
$lines
