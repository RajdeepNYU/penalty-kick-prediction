"""Common time grid (timegrid.py) through P4's kick chain, in a separate cache dir cache_g (Q4P_CACHE) with Q4_GRID=1: kin2b A/F/G, kinR A/F/G, kinMB, seq2b A/B/E/F/G plain and RM;
trees {S}_b2R and NN bags b2 / b2RM on EF and G, seed sets from argv (default 0,1,2). ball2 and camera2 are copies of the main cache; nothing there changes.
usage: python pipeline_grid.py [features] [0,1,2] [3,4,5] -> logs/pipeline_grid.log"""
import os, sys, shutil, threading, time
from concurrent.futures import ThreadPoolExecutor
from runner import P, sh as _sh
LOG = open(f"{P}/logs/pipeline_grid.log", "a", buffering=1); sh = lambda a, env=None: _sh(LOG, a, env)
PC0, GC = f"{P}/cache", f"{P}/cache_g"; E = dict(Q4P_CACHE=GC, Q4_GRID="1"); TH = dict(E, OMP_NUM_THREADS="8", LOKY_MAX_CPU_COUNT="8")
sds = [a for a in sys.argv[1:] if "," in a] or []
os.makedirs(GC, exist_ok=True)
for f in ("ball2.pkl", "camera2.pkl"):
    if not os.path.exists(f"{GC}/{f}"): shutil.copy(f"{PC0}/{f}", f"{GC}/{f}")
if not os.path.isdir(f"{GC}/mb3d"): shutil.copytree(f"{PC0}/mb3d", f"{GC}/mb3d")
if "features" in sys.argv:
    for S in "AFG": sh(["kin2b.py", S, "ball2"], E); sh(["kinR.py", S], E)
    sh(["kinMB.py"], E)
    for S in "ABEFG": sh(["seq2b.py", S], E); sh(["seq2b.py", S, "RM"], E)
    LOG.write(f"[{time.strftime('%H:%M:%S')}] features done\n")


SF = lambda sd: "" if sd == "0,1,2" else "_sd345"


def cpu():
    for sd in sds:
        for S in "AFG":
            if not (os.path.exists(f"{GC}/oof_hgb_{S}_b2R{SF(sd)}.npy") and os.path.exists(f"{GC}/oof_et_{S}_b2R{SF(sd)}.npy")): sh(["e_trees.py", f"{S}_b2R" + SF(sd), f"kin2_{S}_ball2", f"kinR_{S}", f"--seeds={sd}"], TH)


th = threading.Thread(target=cpu); th.start()
cfg = [(12, 96), (15, 128), (20, 96), (20, 128), (30, 128)]
jobs = [(e, h, st, sq, sd) for sd in sds for sq in ("b2", "b2RM") for st in ("EF", "G") for e, h in cfg if not os.path.exists(f"{GC}/oofP_{e}_{h}_st{st}_{sq}{SF(sd)}.npy")]
with ThreadPoolExecutor(3) as ex: list(ex.map(lambda j: sh(["train_seqP.py", str(j[0]), str(j[1])], dict(E, SEQ=j[3], STREAMS=j[2], SEEDS=j[4], OMP_NUM_THREADS="2")), jobs))
th.join(); LOG.write(f"[{time.strftime('%H:%M:%S')}] pipeline_grid done {sds}\n")
