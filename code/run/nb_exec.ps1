# Executes Q4_Penalty_Kick_final.ipynb (nbconvert, no timeout) -> Q4_Penalty_Kick_final_executed.ipynb, with a memory watcher: free commit memory is checked
# every second (logged every 5 min) and the run is stopped if it falls below 3 GB (a stop is better than a crashed PC). Progress: nbrun/logs/notebook_run.log.
# usage: powershell -File nb_exec.ps1
$P = "C:\path\to\project-root\Q4_word_Prime"; Set-Location $P; $L = "$P\nbrun\logs\nb_exec.log"
$py = "python"
"{0} start" -f (Get-Date -Format "MM-dd HH:mm:ss") | Out-File -Append -Encoding utf8 $L
$p = Start-Process $py -ArgumentList "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "Q4_Penalty_Kick_final.ipynb", "--output", "Q4_Penalty_Kick_final_executed.ipynb", "--ExecutePreprocessor.timeout=-1" `
     -PassThru -NoNewWindow -RedirectStandardOutput "$P\nbrun\logs\nbconvert_out.txt" -RedirectStandardError "$P\nbrun\logs\nbconvert_err.txt"
$null = $p.Handle; try { $p.PriorityClass = 'BelowNormal' } catch {}   # the kernel and its script subprocesses inherit below-normal priority
$min = 1e9; $n = 0
while (-not $p.HasExited) {
    $free = (Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory / 1MB; if ($free -lt $min) { $min = $free }
    if ($free -lt 3) { & taskkill /PID $p.Id /T /F | Out-Null; "{0} STOPPED: free commit {1:N1} GB" -f (Get-Date -Format "HH:mm:ss"), $free | Out-File -Append -Encoding utf8 $L }
    if ($n % 300 -eq 0) { "{0} free commit {1:N1} GB (lowest so far {2:N1})" -f (Get-Date -Format "HH:mm:ss"), $free, $min | Out-File -Append -Encoding utf8 $L }
    $n++; Start-Sleep 1
}
"{0} done, exit {1} | lowest free commit {2:N1} GB" -f (Get-Date -Format "HH:mm:ss"), $p.ExitCode, $min | Out-File -Append -Encoding utf8 $L
"exit $($p.ExitCode), lowest free commit {0:N1} GB" -f $min
