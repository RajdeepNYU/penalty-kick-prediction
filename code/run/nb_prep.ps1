# Seeds the notebook's clean-room folder nbrun/ with copies of the caches of the original run (read-only on the sources; nothing is deleted or moved).
#   nbrun/wcache   <- Q4_work/cache (perception: detections, tracks, five kicker pose streams, keeper pose, keeper CNN)
#   nbrun/cache    <- Q4_word_Prime/cache, without the experiment crops / embeddings the final recipe never reads (ce_*, crops*, zoom*, foot_emb),
#                     then the ones it does read: crops8 / crops5 (+ meta), zoom3_meta, cropsfix_D, ViTPose+ base tokens of body_hm / body_tor / legs12 (+ D)
#   nbrun/cache_kf, nbrun/cache_g <- the re-pick chain and the 25 fps grid rows; nbrun/models <- MotionBERT, NBJW, WASB (code + weights)
# With empty caches every notebook stage runs from the raw videos instead. usage: powershell -File nb_prep.ps1   -> nbrun/logs/prep.log
$ErrorActionPreference = "Stop"
$P = "C:\path\to\project-root\Q4_word_Prime"; $Q = "C:\path\to\project-root\Q4_work"; $R = "$P\nbrun"
New-Item -ItemType Directory -Force "$R\logs" | Out-Null; $L = "$R\logs\prep.log"
function rc($src, $dst, [string[]]$extra) {
    $t0 = Get-Date; & robocopy $src $dst @extra /E /COPY:DAT /DCOPY:T /R:1 /W:1 /MT:8 /NFL /NDL /NP /LOG+:$L | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy $src failed ($LASTEXITCODE)" }
    "{0} {1} -> {2} ({3:N0}s, rc {4})" -f (Get-Date -Format HH:mm:ss), $src, $dst, ((Get-Date) - $t0).TotalSeconds, $LASTEXITCODE
}
rc "$Q\cache" "$R\wcache" @()
rc "$P\cache" "$R\cache" @("/XF", "ce_*", "crops*", "zoom*", "foot_emb.npz")
$keep = @("crops8.npy", "crops8_meta.npz", "crops5.npy", "crops5_meta.npz", "zoom3_meta.npz", "cropsfix_D.npz") + (@("body_hm", "body_tor", "legs12") | ForEach-Object { "ce_$($_)_vpbase.npy"; "ce_$($_)_vpbaseD.npy" })
$t0 = Get-Date; & robocopy "$P\cache" "$R\cache" @keep /COPY:DAT /R:1 /W:1 /NFL /NDL /NP /LOG+:$L | Out-Null; if ($LASTEXITCODE -ge 8) { throw "robocopy keep failed" }
"{0} cache: {1} read files ({2:N0}s)" -f (Get-Date -Format HH:mm:ss), $keep.Count, ((Get-Date) - $t0).TotalSeconds
rc "$P\cache_kf" "$R\cache_kf" @()
rc "$P\cache_g" "$R\cache_g" @()
foreach ($m in "motionbert", "nbjw", "nbjw_src", "wasb", "wasb_src") { rc "$P\models\$m" "$R\models\$m" @() }
foreach ($d in "wcache", "cache", "cache_kf", "cache_g", "models") { $s = (Get-ChildItem "$R\$d" -Recurse -File | Measure-Object Length -Sum); "{0,-9} {1,7:N2} GB {2,7} files" -f $d, ($s.Sum / 1GB), $s.Count }
