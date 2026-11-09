"""Idea 3a (RULES_nov4.md): camera angle below the horizon (sin phi < 0). If camera2's goal side (sign of cos phi) disagrees with the calibration flip, the goal direction is reversed: phi + 180; else phi -> -phi.
make  -> cache/camera2f.pkl (camera2 with the flagged clips refitted from (s, theta, phi')), then kinR A/F/G with CAMF=camera2f -> cache/kinR_{S}_camera2f.pkl
trees -> per partition set: trees trained on the original rows (as e_trees.py), read on original and corrected test rows -> cache/nov4/cam_trees_{set}.npz
usage: python cap.py nov4_cam.py make | trees 012|345|91011"""
import os, sys, pickle, subprocess, numpy as np
from common import *

FLP = np.load(f"{PC}/zoom3_meta.npz"); FLIP = dict(zip(list(FLP["clips"]), FLP["flip"]))


def model(p):  # camera2.py's model; camera2.py itself is not imported, since importing it refits and rewrites camera2.pkl
    s, th, ph = p; st, ct = np.sin(th), np.cos(th)
    return np.array([s * np.cos(ph), -s * st * np.sin(ph)]), np.array([-s * np.sin(ph), -s * st * np.cos(ph)]), np.array([0.0, -s * ct])


def fixed_phi(d, c):
    ph = np.radians(d["phi"])
    if np.sin(ph) >= 0: return None
    goal_right = np.cos(ph) > 0
    return float(np.degrees(ph + np.pi)) if goal_right == bool(FLIP[c]) else float(-d["phi"])      # flip True = goal on the left, so goal_right == flip means they disagree


if sys.argv[1] == "make":
    CAM = pickle.load(open(f"{PC}/camera2.pkl", "rb")); out = dict(CAM); prv = set(pm["clip"]); n = 0
    for c, d in CAM.items():
        if d is None or "phi" not in d: continue
        p2 = fixed_phi(d, c)
        if p2 is None: continue
        e = dict(d); mu, mv, mz = model([d["s"], np.radians(d["theta"]), np.radians(p2)]); e.update(phi=p2, A=np.c_[mu, mv], az=mz, phi_old=d["phi"]); e["ball_ground"] = e["ball"] - e["az"] * 0.11; e["Ai"] = np.linalg.inv(e["A"]); e["keeper_g"] = e["Ai"] @ (e["kfeet"] - e["ball_ground"]); out[c] = e; n += 1
        print(f"   {c:18s} {'test' if c in prv else 'labelled':8s} flip {int(FLIP[c])} phi {d['phi']:7.1f} -> {p2:7.1f} | {'reversed' if abs(((p2 - d['phi']) % 360) - 180) < 1e-6 else 'vertical'}", flush=True)
    pickle.dump(out, open(f"{PC}/camera2f.pkl", "wb")); print("flagged", n, flush=True)
    for S in "AFG": subprocess.run([sys.executable, "kinR.py", S], env=dict(os.environ, CAMF="camera2f"), check=True)

if sys.argv[1] == "trees":
    from cvlib import hg, et, stack, yk, ym, grp, cl, SW
    from sklearn.model_selection import StratifiedGroupKFold
    SET = sys.argv[2]; SEEDS = {"012": (0, 1, 2), "345": (3, 4, 5), "91011": (9, 10, 11)}[SET]

    def feats(kr):
        Xa, Xb, nm = [], [], []
        for f in kr:
            K = pickle.load(open(f"{PC}/{f}.pkl", "rb")); r = K["rows"]; nm += list(K["names"])
            if isinstance(next(iter(r.values()))[0], tuple): r = {c: (v[0][0], v[1][0]) for c, v in r.items()}
            a, b = stack(r, gt["clip"], len(K["names"])); Xa.append(a); Xb.append(b)
        return np.concatenate(Xa, 1), np.concatenate(Xb, 1), nm

    R = {k: np.zeros((len(gt), 3)) for k in ("orig", "cam")}
    for S in "AFG":
        V = dict(orig=feats([f"kin2_{S}_ball2", f"kinR_{S}"]), cam=feats([f"kin2_{S}_ball2", f"kinR_{S}_camera2f"])); assert V["orig"][2] == V["cam"][2]; keep = ~np.isnan(V["orig"][0]).all(0); V = {k: (a[:, keep], b[:, keep]) for k, (a, b, _) in V.items()}; Xa, Xb = V["orig"]
        ch = np.where(~(np.isclose(V["orig"][0], V["cam"][0], equal_nan=True).all(1)))[0]; print(S, "rows changed by the corrected camera:", len(ch), flush=True)
        for mk in (hg, et):
            for s in SEEDS:
                for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(Xa, yk, grp):
                    m = mk(s).fit(np.r_[Xa[tr], Xb[tr]], np.r_[yk[tr], ym[tr]]); c_ = list(m.classes_); ix = [c_.index(SW[k]) for k in c_]; o_ = [c_.index(k) for k in cl]
                    for k, (a, b) in V.items(): R[k][te] += ((m.predict_proba(a[te]) + m.predict_proba(b[te])[:, ix]) / 2)[:, o_] / len(SEEDS) / 2 / 3
    os.makedirs(f"{PC}/nov4", exist_ok=True); np.savez(f"{PC}/nov4/cam_trees_{SET}.npz", **R); print("saved", flush=True)
