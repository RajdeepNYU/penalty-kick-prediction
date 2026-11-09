"""Chain 34: the kicker re-pick (kfix.py) through P4's kick chain, in a separate cache dir cache_kf (Q4P_CACHE) with the per-clip overlay cache/kfixW (Q4_OVL):
RTMW3D (B stream) for the re-picked clips, dense ViTPose-L + MotionBERT (only the re-picked clips are missing in the copied kptsD / mb3d), ball2 for the
re-picked clips merged into a copy of ball2.pkl, then camera2, kin2b A/F/G, kinR A/F/G, kinMB, seq2b A/B/E/F/G plain and RM over all clips (fast);
trees {S}_b2R and NN bags b2 / b2RM on EF and G, seed sets 0,1,2 (then 3,4,5). OOFs land in cache_kf. usage: python pipeline34.py -> logs/pipeline34.log"""
import os, sys, shutil, pickle, time, threading
from concurrent.futures import ThreadPoolExecutor
from runner import P, sh as _sh
LOG = open(f"{P}/logs/pipeline34.log", "a", buffering=1); sh = lambda a, env=None: _sh(LOG, a, env)
PC0, KC = f"{P}/cache", f"{P}/cache_kf"; KF = set(pickle.load(open(f"{PC0}/kfix_clips.pkl", "rb"))); E = dict(Q4P_CACHE=KC, Q4_OVL=f"{PC0}/kfixW")
TH = dict(E, OMP_NUM_THREADS="8", LOKY_MAX_CPU_COUNT="8")
os.makedirs(KC, exist_ok=True)
for f in ("goalposts.pkl", "ball2.pkl"):
    if not os.path.exists(f"{KC}/{f}"): shutil.copy(f"{PC0}/{f}", f"{KC}/{f}")
for d in ("kptsD", "mb3d"):
    os.makedirs(f"{KC}/{d}", exist_ok=True)
    for fn in os.listdir(f"{PC0}/{d}"):
        if fn[:-4] not in KF and not os.path.exists(f"{KC}/{d}/{fn}"): shutil.copy(f"{PC0}/{d}/{fn}", f"{KC}/{d}/{fn}")
LOG.write(f"[{time.strftime('%H:%M:%S')}] cache_kf ready, re-picked clips {len(KF)}\n")
if "resume" not in sys.argv:                                               # resume: after the camera2 patch (non-re-picked clips keep cache/camera2.pkl)
    while not (os.path.isdir(f"{PC0}/kfixW/kptsG") and len(os.listdir(f"{PC0}/kfixW/kptsG")) >= len(KF)): time.sleep(20)     # kfix.py pose (run separately) finished
    sh(["kfix.py", "pose3d"])
    sh(["kpts_dense.py"], E); sh(["-I", f"{P}/lift_mb.py"], E)
    src = open(f"{P}/ball2.py", encoding="utf-8").read()                   # ball2 only for the re-picked clips, merged into the copy
    src = "import sys; sys.path.insert(0, r'%s')\n" % P + src.replace("for n, c in enumerate(CLIPS):", f"for n, c in enumerate([c for c in CLIPS if c in {sorted(KF)!r}]):", 1)
    src = src.replace('pickle.dump(out, open(f"{PC}/ball2.pkl", "wb"))', 'full = pickle.load(open(f"{PC}/ball2.pkl", "rb")); full.update(out); pickle.dump(full, open(f"{PC}/ball2.pkl", "wb"))', 1)
    assert "full.update(out)" in src; open(f"{KC}/_ball2_kf.py", "w", encoding="utf-8").write(src); sh([f"{KC}/_ball2_kf.py"], E)
    sh(["camera2.py"], E)                                                  # its no-posts prior moves 115 other clips by ~0.1 deg: patched by hand, then resume
for S in "AFG": sh(["kin2b.py", S, "ball2"], E); sh(["kinR.py", S], E)
sh(["kinMB.py"], E)
for S in "ABEFG": sh(["seq2b.py", S], E); sh(["seq2b.py", S, "RM"], E)
LOG.write(f"[{time.strftime('%H:%M:%S')}] features done\n")


def cpu():
    for sd in ("0,1,2", "3,4,5"):
        for S in "AFG": sh(["e_trees.py", f"{S}_b2R" + ("" if sd == "0,1,2" else "_sd345"), f"kin2_{S}_ball2", f"kinR_{S}", f"--seeds={sd}"], TH)


th = threading.Thread(target=cpu); th.start()
cfg = [(12, 96), (15, 128), (20, 96), (20, 128), (30, 128)]
jobs = [(e, h, st, sq, sd) for sd in ("0,1,2", "3,4,5") for sq in ("b2", "b2RM") for st in ("EF", "G") for e, h in cfg]
with ThreadPoolExecutor(3) as ex: list(ex.map(lambda j: sh(["train_seqP.py", str(j[0]), str(j[1])], dict(E, SEQ=j[3], STREAMS=j[2], SEEDS=j[4], OMP_NUM_THREADS="2")), jobs))
th.join(); LOG.write(f"[{time.strftime('%H:%M:%S')}] pipeline34 done\n")
