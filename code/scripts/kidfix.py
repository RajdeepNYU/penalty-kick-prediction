"""Kicker-identity fix (kid.py, variant 'G* mono 0.2 + re-pick motion': motion-gated tracking from the anchor pick with the no-retreat gate, re-pick of the
fastest approacher near the spot when the track still ends > 6 m from it, low-confidence TTA persons for that re-pick) through P4's kick chain.
Only clips whose kicker track changes vs the current overlay (some timed frame IoU < 0.5) are rebuilt; every other clip keeps its current files.
  prep : cache/kidW overlay (tracks.pkl = current tracks with the changed clips replaced; per-clip files of the kfix overlay for unchanged re-picked
         clips; det copies for the changed clips) and cache_kid (copy of cache_kf without the changed clips' dense pose / MotionBERT) + cache/kid_clips.pkl
  pose : ballspot, RTMW, ViTPose E/F/G for the changed clips (nb_src scripts with Q4_CACHE=cache/kidW)
  rest : dense ViTPose + MotionBERT, ball2 (merged), camera2 (other clips patched back), kin2b / kinR / kinMB / seq2b, trees and NN bags (2 seed sets)
usage: python kidfix.py prep | pose | rest   -> logs/kidfix.log"""
import sys, shutil, subprocess, time, threading
from concurrent.futures import ThreadPoolExecutor
from common import *
from runner import sh as _sh

VAR = "G* mono 0.2 + re-pick motion"; OV, OVk, KC0, KC = f"{PC}/kidW", f"{PC}/kfixW", f"{P}/cache_kf", f"{P}/cache_kid"
LOG = open(f"{P}/logs/kidfix.log", "a", buffering=1); sh = lambda a, env=None: _sh(LOG, a, env)
E = dict(Q4P_CACHE=KC, Q4_OVL=OV); TH = dict(E, OMP_NUM_THREADS="8", LOKY_MAX_CPU_COUNT="8")
SUBS = ("ballspot", "det", "kpts", "kpts3d", "kptsE", "kptsF", "kptsG")
stage = sys.argv[1]

if stage == "prep":
    T1 = pickle.load(open(f"{OVk}/tracks.pkl", "rb")); K = pickle.load(open(f"{PC}/kid_combo.pkl", "rb"))[VAR]; KF = set(pickle.load(open(f"{PC}/kfix_clips.pkl", "rb")))
    def iou(a, b):
        ix = np.maximum(0, np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0])); iy = np.maximum(0, np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1]))
        i = ix * iy; return i / np.maximum((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - i, 1e-6)
    CH = sorted(c for c in K if T1.get(c) is not None and np.nanmin(iou(np.asarray(T1[c]["kick"], float), K[c]["kick"])) < 0.5)
    T = dict(T1)
    for c in CH: T[c] = dict(T1[c], kick=np.asarray(K[c]["kick"], float), kick_a=np.asarray(K[c]["kick_a"], float))
    for s in SUBS: os.makedirs(f"{OV}/{s}", exist_ok=True)
    pickle.dump(T, open(f"{OV}/tracks.pkl", "wb")); pickle.dump(CH, open(f"{PC}/kid_clips.pkl", "wb"))
    for c in KF - set(CH):                                                  # unchanged re-picked clips keep the kfix overlay files
        for s in SUBS: shutil.copy(f"{OVk}/{s}/{c}.pkl", f"{OV}/{s}/{c}.pkl")
    for c in CH: shutil.copy(f"{W}/det/{c}.pkl", f"{OV}/det/{c}.pkl")
    os.makedirs(KC, exist_ok=True)
    for f in ("goalposts.pkl", "ball2.pkl", "camera2.pkl"): shutil.copy(f"{KC0}/{f}", f"{KC}/{f}")
    for d in ("kptsD", "mb3d"):
        os.makedirs(f"{KC}/{d}", exist_ok=True)
        for fn in os.listdir(f"{KC0}/{d}"):
            if fn[:-4] not in CH: shutil.copy(f"{KC0}/{d}/{fn}", f"{KC}/{d}/{fn}")
    print(f"changed clips {len(CH)} (public {sum(SPLIT[c] == 'train' for c in CH)}, private {sum(SPLIT[c] == 'val' for c in CH)}); of them kfix re-picked {len(set(CH) & KF)}")

elif stage in ("pose", "pose3d"):                                           # pose3d: RTMW3D (stream B of the NN), missed by the first run
    CH = pickle.load(open(f"{PC}/kid_clips.pkl", "rb")); env = dict(os.environ, Q4_CACHE=OV)
    for scr, ex in (("ballspot.py", {}), ("kpts.py", {}), ("kpts_vit.py", dict(VIT_OUT="kptsE", VIT_MODEL="usyd-community/vitpose-plus-base")),
                    ("kpts_vit.py", dict(VIT_OUT="kptsF", VIT_MODEL="usyd-community/vitpose-plus-large")), ("kpts_vit.py", dict(VIT_OUT="kptsG", VIT_MODEL="usyd-community/vitpose-plus-huge")))\
            if stage == "pose" else (("kpts3d.py", {}),):
        src = open(f"{P}/nb_src/{scr}", encoding="utf-8").read(); assert "for split, c in todo:" in src
        src = src.replace("for split, c in todo:", f"todo = [t for t in todo if t[1] in {CH!r}]\nfor split, c in todo:", 1)
        fp = f"{OV}/_run_{scr}"; open(fp, "w", encoding="utf-8").write(src); t0 = time.time()
        r = subprocess.run([sys.executable, fp], env=dict(env, **ex), cwd=f"{P}/nb_src"); LOG.write(f"[{time.strftime('%H:%M:%S')}] pose {scr} {ex.get('VIT_OUT', '')} rc {r.returncode} ({time.time() - t0:.0f}s)\n")
        if r.returncode: sys.exit(1)

elif stage == "rest":
    CH = set(pickle.load(open(f"{PC}/kid_clips.pkl", "rb")))
    sh(["kpts_dense.py"], E); sh(["-I", f"{P}/lift_mb.py"], E)
    src = open(f"{P}/ball2.py", encoding="utf-8").read()                   # ball2 only for the changed clips, merged into the copy
    src = "import sys; sys.path.insert(0, r'%s')\n" % P + src.replace("for n, c in enumerate(CLIPS):", f"for n, c in enumerate([c for c in CLIPS if c in {sorted(CH)!r}]):", 1)
    src = src.replace('pickle.dump(out, open(f"{PC}/ball2.pkl", "wb"))', 'full = pickle.load(open(f"{PC}/ball2.pkl", "rb")); full.update(out); pickle.dump(full, open(f"{PC}/ball2.pkl", "wb"))', 1)
    assert "full.update(out)" in src; open(f"{KC}/_ball2_kid.py", "w", encoding="utf-8").write(src); sh([f"{KC}/_ball2_kid.py"], E)
    sh(["camera2.py"], E)                                                  # its no-posts prior shifts other clips when any clip changes: put them back
    new, old = pickle.load(open(f"{KC}/camera2.pkl", "rb")), pickle.load(open(f"{KC0}/camera2.pkl", "rb"))
    for c in old:
        if c not in CH: new[c] = old[c]
    pickle.dump(new, open(f"{KC}/camera2.pkl", "wb")); LOG.write(f"[{time.strftime('%H:%M:%S')}] camera2 patched back for {len(old) - len(CH)} unchanged clips\n")
    for S in "AFG": sh(["kin2b.py", S, "ball2"], E); sh(["kinR.py", S], E)
    sh(["kinMB.py"], E)
    for S in "EFG": sh(["seq2b.py", S], E); sh(["seq2b.py", S, "RM"], E)
    LOG.write(f"[{time.strftime('%H:%M:%S')}] features done\n")

    def cpu():
        for sd in ("0,1,2", "3,4,5"):
            for S in "AFG": sh(["e_trees.py", f"{S}_b2R" + ("" if sd == "0,1,2" else "_sd345"), f"kin2_{S}_ball2", f"kinR_{S}", f"--seeds={sd}"], TH)
        LOG.write(f"[{time.strftime('%H:%M:%S')}] trees done\n")
    th = threading.Thread(target=cpu); th.start()
    cfg = [(12, 96), (15, 128), (20, 96), (20, 128), (30, 128)]
    jobs = [(e, h, st, sq, sd) for sd in ("0,1,2", "3,4,5") for sq in ("b2", "b2RM") for st in ("EF", "G") for e, h in cfg]
    with ThreadPoolExecutor(3) as ex: list(ex.map(lambda j: sh(["train_seqP.py", str(j[0]), str(j[1])], dict(E, SEQ=j[3], STREAMS=j[2], SEEDS=j[4], OMP_NUM_THREADS="2")), jobs))
    th.join(); LOG.write(f"[{time.strftime('%H:%M:%S')}] kidfix done\n")

elif stage == "nn":                                                         # rerun of the NN bags after pose3d: base streams A, B of train_seqP
    for S in "AB": sh(["seq2b.py", S], E); sh(["seq2b.py", S, "RM"], E)
    cfg = [(12, 96), (15, 128), (20, 96), (20, 128), (30, 128)]
    jobs = [(e, h, st, sq, sd) for sd in ("0,1,2", "3,4,5") for sq in ("b2", "b2RM") for st in ("EF", "G") for e, h in cfg]
    with ThreadPoolExecutor(3) as ex: list(ex.map(lambda j: sh(["train_seqP.py", str(j[0]), str(j[1])], dict(E, SEQ=j[3], STREAMS=j[2], SEEDS=j[4], OMP_NUM_THREADS="2")), jobs))
    LOG.write(f"[{time.strftime('%H:%M:%S')}] kidfix nn done\n")
