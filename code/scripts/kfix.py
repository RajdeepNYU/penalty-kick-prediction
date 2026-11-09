"""Kicker re-pick for clips whose tracked kicker is not the player at the ball (kickcheck.py: tracked box > 6 m from the NBJW-projected penalty spot on the
ground plane, another detected person < 2 m from it). The new kicker is that person at the last det frame, followed back over the 13 timed frames with
Q4_work/track5.py's association (det persons conf >= 0.1, centre/height cost < 1.5). Label-free; the same rule on public and private clips.
Writes the overlay dir cache/kfixW (read by common.load / tracks() when Q4_OVL points there): tracks.pkl (full copy, re-picked kick + kick_a), det/<clip>.pkl
copies for the re-picked clips (Q4_work/cache is never written), and cache/kfix_clips.pkl. Then:
  stage pose: the nb_src per-clip scripts (ballspot, kpts, kpts_vit E/F/G) restricted to these clips with Q4_CACHE=cache/kfixW
usage: python kfix.py pick | python kfix.py pose"""
import sys, shutil, subprocess, numpy as np
from common import *

OV = f"{PC}/kfixW"; os.makedirs(f"{OV}/det", exist_ok=True)


def cost_to(prev, bb):
    hp = max(prev[3] - prev[1], 1); cp = np.array([(prev[0] + prev[2]) / 2, (prev[1] + prev[3]) / 2]); cc = np.c_[(bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2]
    return np.linalg.norm(cc - cp, axis=1) / hp + 1.5 * np.abs(np.log(np.maximum((bb[:, 3] - bb[:, 1]) / hp, 1e-3)))


def follow(d, start, a0, thr=0.1):
    n = len(d["frames"]); out = np.full((n, 4), np.nan); out[a0] = start; prev = start.copy(); j = a0 - 1
    while j >= 0:
        f = d["frames"][j]; bb = np.asarray(f["pb"], float)[np.asarray(f["pc"]) >= thr]
        if len(bb):
            cst = cost_to(prev, bb); i = cst.argmin()
            if cst[i] < 1.5: out[j] = bb[i]; prev = bb[i]
        j -= 1
    return out


if sys.argv[1] == "pick":
    KC = pickle.load(open(f"{PC}/kickcheck.pkl", "rb")); T = pickle.load(open(f"{W}/tracks.pkl", "rb")); KF = []
    for c in CLIPS:
        r = KC.get(c)
        if r is None or not (r["d_track"] > 6 and r["d_best"] < 2 and r["iou"] < 0.3): continue
        d = load("det", c); kt = follow(d, np.asarray(r["best_box"], float), len(d["frames"]) - 1)
        T[c] = dict(T[c], kick=kt, kick_a=kt[6] if np.isfinite(kt[6, 0]) else kt[-1]); KF.append(c); shutil.copy(f"{W}/det/{c}.pkl", f"{OV}/det/{c}.pkl")
        print(c, "frames followed", int(np.isfinite(kt[:, 0]).sum()), "of", len(kt))
    pickle.dump(T, open(f"{OV}/tracks.pkl", "wb")); pickle.dump(KF, open(f"{PC}/kfix_clips.pkl", "wb"))
    print("re-picked", len(KF), "clips: public", sum(c in SPLIT and SPLIT[c] == "train" for c in KF), "private", sum(SPLIT[c] == "val" for c in KF))
elif sys.argv[1] in ("pose", "pose3d"):
    KF = set(pickle.load(open(f"{PC}/kfix_clips.pkl", "rb"))); env = dict(os.environ, Q4_CACHE=OV)
    for scr, ex in (("ballspot.py", {}), ("kpts.py", {}), ("kpts_vit.py", dict(VIT_OUT="kptsE", VIT_MODEL="usyd-community/vitpose-plus-base")),
                    ("kpts_vit.py", dict(VIT_OUT="kptsF", VIT_MODEL="usyd-community/vitpose-plus-large")), ("kpts_vit.py", dict(VIT_OUT="kptsG", VIT_MODEL="usyd-community/vitpose-plus-huge")))\
            if sys.argv[1] == "pose" else (("kpts3d.py", {}),):
        src = open(f"{P}/nb_src/{scr}", encoding="utf-8").read(); assert "for split, c in todo:" in src
        src = src.replace("for split, c in todo:", f"todo = [t for t in todo if t[1] in {sorted(KF)!r}]\nfor split, c in todo:", 1)
        fp = f"{OV}/_run_{scr}"; open(fp, "w", encoding="utf-8").write(src); t0 = time.time() if "time" in globals() else 0
        r = subprocess.run([sys.executable, fp], env=dict(env, **ex), cwd=f"{P}/nb_src"); print(scr, ex.get("VIT_OUT", ""), "rc", r.returncode, flush=True)
        if r.returncode: sys.exit(1)
