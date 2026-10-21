# Run a measurement program and sample its GPU memory (#458, step E1).
#
# Windows reports the GPU memory of each process in the "GPU Process Memory" performance counters
# (the values that Task Manager shows). The script starts the program, samples the dedicated and
# shared usage of its process and of all its child processes every -IntervalMs, and writes the
# peak values and the samples to -Out. The program output goes to -Log. The child processes count
# because a Python virtual environment `python.exe` is a launcher that runs the interpreter as a
# child process.
#
# Example:
#   .\run_with_gpu_memory.ps1 -Exe ..\..\..\rust\puerh_mind\target\release\examples\segmentation_runtime_report.exe `
#     -Arguments @('runtime', '--onnx-dir', 'D:\...\onnx\tiny', '--device', 'directml', ...) `
#     -Out gpu_memory_tiny.json -Log runtime_tiny.log

param(
    [Parameter(Mandatory = $true)][string]$Exe,
    [Parameter(Mandatory = $true)][string[]]$Arguments,
    [Parameter(Mandatory = $true)][string]$Out,
    [Parameter(Mandatory = $true)][string]$Log,
    [int]$IntervalMs = 250,
    [int]$TreeRefreshSeconds = 10
)

$ErrorActionPreference = 'Stop'

function Quote([string]$Value) {
    if ($Value -match '[\s"]') { return '"' + ($Value -replace '"', '\"') + '"' }
    return $Value
}

$process = Start-Process -FilePath $Exe -ArgumentList (($Arguments | ForEach-Object { Quote $_ }) -join ' ') `
    -NoNewWindow -PassThru -RedirectStandardOutput $Log -RedirectStandardError "$Log.stderr"
$samples = New-Object System.Collections.Generic.List[object]
$started = Get-Date
$peakDedicated = 0.0
$peakShared = 0.0
function Get-ProcessTreeIds([int]$RootId) {
    $children = @{}
    foreach ($item in Get-CimInstance Win32_Process -Property ProcessId, ParentProcessId) {
        $children[[int]$item.ParentProcessId] += @([int]$item.ProcessId)
    }
    $ids = New-Object System.Collections.Generic.List[int]
    $queue = New-Object System.Collections.Generic.Queue[int]
    $queue.Enqueue($RootId)
    while ($queue.Count -gt 0) {
        $id = $queue.Dequeue()
        if ($ids.Contains($id)) { continue }
        $ids.Add($id)
        foreach ($child in @($children[$id])) { if ($null -ne $child) { $queue.Enqueue($child) } }
    }
    return $ids
}

# The process tree query takes about 1.5 s. It runs in each loop until a process of the tree
# opens the GPU, then every -TreeRefreshSeconds.
$treeIds = @()
$treeRead = [datetime]::MinValue
while (-not $process.HasExited) {
    try {
        if (((Get-Date) - $treeRead).TotalSeconds -ge $TreeRefreshSeconds -or $samples.Count -eq 0) {
            $treeIds = @(Get-ProcessTreeIds $process.Id)
            $treeRead = Get-Date
        }
        $counters = foreach ($id in $treeIds) {
            "\GPU Process Memory(pid_$($id)_*)\Dedicated Usage"
            "\GPU Process Memory(pid_$($id)_*)\Shared Usage"
        }
        $values = (Get-Counter -Counter $counters -ErrorAction SilentlyContinue).CounterSamples
        if (-not $values) { throw 'no counter instance yet' }
        $dedicated = ($values | Where-Object { $_.Path -like '*dedicated usage' } | Measure-Object CookedValue -Sum).Sum
        $shared = ($values | Where-Object { $_.Path -like '*shared usage' } | Measure-Object CookedValue -Sum).Sum
        $peakDedicated = [Math]::Max($peakDedicated, $dedicated)
        $peakShared = [Math]::Max($peakShared, $shared)
        $samples.Add([ordered]@{
                seconds        = [Math]::Round(((Get-Date) - $started).TotalSeconds, 2)
                dedicated_bytes = $dedicated
                shared_bytes   = $shared
            })
    }
    catch {
        # The counter instance appears only after a process of the tree opens the GPU.
    }
    Start-Sleep -Milliseconds $IntervalMs
}
$process.WaitForExit()

$report = [ordered]@{
    source               = 'Windows performance counters: GPU Process Memory (Dedicated Usage, Shared Usage)'
    exit_code            = $process.ExitCode
    interval_ms          = $IntervalMs
    peak_dedicated_bytes = $peakDedicated
    peak_shared_bytes    = $peakShared
    sample_count         = $samples.Count
    samples              = $samples
}
# LF line endings (ConvertTo-Json writes CRLF on Windows).
$json = ($report | ConvertTo-Json -Depth 4) -replace "`r`n", "`n"
[System.IO.File]::WriteAllText($ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Out), $json + "`n", [System.Text.UTF8Encoding]::new($false))
Write-Output ("exit {0}; peak dedicated {1:N0} MB; peak shared {2:N0} MB; {3} samples" -f `
        $process.ExitCode, ($peakDedicated / 1MB), ($peakShared / 1MB), $samples.Count)
exit $process.ExitCode
