"""Kicker identity experiments (read-only on every cache; writes cache/kid_*.pkl). Compares, on the kid.py harness:
T0 original tracks, T1 current (kfix re-pick), G gated re-track from the anchor pick, SPL splice of T0 / T1 at the crossing, and re-pick variants.
usage: python kid_exp.py base"""
import sys, time
from kid import *

T0 = pickle.load(open(f"{W}/tracks.pkl", "rb")); T1 = pickle.load(open(f"{PC}/kfixW/tracks.pkl", "rb")); KF = set(pickle.load(open(f"{PC}/kfix_clips.pkl", "rb")))
stage = sys.argv[1]

if stage == "base":
    t0 = time.time(); G = {}; nm = {}
    for i, c in enumerate(CLIPS):
        g = geo(c)
        if g is None or T0.get(c) is None: continue
        a = np.asarray(T0[c]["kick_a"], float); kb, m = gtrack(g, ANCHOR, a); G[c] = dict(T0[c], kick=kb, kick_a=a); nm[c] = int((~m).sum())
        if i % 500 == 0: print(i, f"{time.time() - t0:.0f}s", flush=True)
    pickle.dump(dict(G=G, nm=nm), open(f"{PC}/kid_G.pkl", "wb"))
    E0 = evaluate(T0, name="T0 original"); E1 = evaluate(T1, name="T1 current (re-pick)"); EG = evaluate(G, name="G gated from anchor")
    print("gated: frames interpolated per clip", pd.Series(nm).describe().round(2).to_dict())
    for nmE, E in (("T0", E0), ("T1", E1), ("G", EG)): pickle.dump(E, open(f"{PC}/kid_E_{nmE}.pkl", "wb"))

elif stage == "motion":                                                    # the user's cue: every person at the anchor, tracked, scored by approach to the ball
    rows = []; t0 = time.time()
    for i, c in enumerate(CLIPS):
        g = geo(c)
        if g is None or T0.get(c) is None or not len(g["P"][ANCHOR]): continue
        pk = np.asarray(T0[c]["kick_a"], float); cp = click_px(c, g) if c in CK.index else None
        for k, b in enumerate(g["P"][ANCHOR]):
            kb, m = gtrack(g, ANCHOR, b, vmax=2, slack=0.6, damp=0.0); ap = approach(g, kb)
            last = kb[-1]; d_ball = np.linalg.norm(foot(wbox(g["H"][-1], last))[0] - g["ball"]) / max(last[3] - last[1], 1) if g["ball"] is not None else np.nan
            ix = max(0, min(b[2], pk[2]) - max(b[0], pk[0])) * max(0, min(b[3], pk[3]) - max(b[1], pk[1])); io = ix / ((b[2] - b[0]) * (b[3] - b[1]) + (pk[2] - pk[0]) * (pk[3] - pk[1]) - ix)
            rows.append(dict(clip=c, k=k, split=SPLIT[c], pick=io > 0.5, click=float(inb(b, cp)) if cp is not None else np.nan, h=b[3] - b[1], cy=b[3],
                             spot_end=spot_m(g, last), d_ball_end=d_ball, nmatch=int(m.sum()), **{f"ap_{q}": v for q, v in ap.items()}))
        if i % 300 == 0: print(i, f"{time.time() - t0:.0f}s", flush=True)
    pickle.dump(pd.DataFrame(rows), open(f"{PC}/kid_motion.pkl", "wb"))

elif stage == "combo":                                                     # gated tracks + re-pick variants for tracks still ending far from the spot
    GS = pickle.load(open(f"{PC}/kid_G_2_0.6_0.0.pkl", "rb")); kw = dict(vmax=2, slack=0.6, damp=0.0)

    def repick(G, far=6.0, near=2.0, how="nearest"):
        out, n = dict(G), []
        for c, t in G.items():
            g = geo(c)
            if g is None or g["gr"] is None: continue
            if not spot_m(g, t["kick"][-1]) > far: continue
            P_ = g["P"][-1]
            if not len(P_): continue
            ds = np.array([spot_m(g, b) for b in P_]); cand = np.where(ds < near)[0]
            if not len(cand): continue
            if how == "nearest": i = cand[ds[cand].argmin()]
            else:                                                          # the user's cue: among people near the spot, the one approaching fastest
                tr = [gtrack(g, len(P_) and len(g["P"]) - 1, P_[i], **kw)[0] for i in cand]; v = np.array([approach(g, kb)["v"] for kb in tr])
                i = cand[np.nanargmax(np.where(np.isfinite(v), v, -np.inf))] if np.isfinite(v).any() else cand[ds[cand].argmin()]
            kb, m = gtrack(g, len(g["P"]) - 1, P_[i], **kw); out[c] = dict(t, kick=kb, kick_a=kb[ANCHOR]); n.append(c)
        return out, n
    R = {}
    for nm, args in (("G* + re-pick nearest (>6 m)", dict()), ("G* + re-pick fastest approacher", dict(how="fastest")),
                     ("G* + re-pick nearest (>3 m)", dict(far=3.0)), ("G* + re-pick nearest (>4.5 m)", dict(far=4.5))):
        R[nm], n = repick(GS, **args); evaluate(R[nm], name=nm); print(f"   re-picked {len(n)} (public {sum(SPLIT[c] == 'train' for c in n)}, private {sum(SPLIT[c] == 'val' for c in n)})", flush=True)
    pickle.dump(R, open(f"{PC}/kid_combo.pkl", "wb"))

elif stage == "tta":                                                       # the missed kicker: low-confidence TTA person detection where nobody stands on the spot
    from ultralytics import YOLO
    R = pickle.load(open(f"{PC}/kid_combo.pkl", "rb"))["G* + re-pick nearest (>6 m)"]; todo = []
    for c, t in R.items():
        g = geo(c)
        if g is None or g["gr"] is None: continue
        s = spot_m(g, t["kick"][-1]); sp = g["spot_px"]
        if 6 < s < 50 and 0 <= sp[0] <= g["wh"][0] and 0 <= sp[1] <= g["wh"][1]: todo.append(c)  # beyond 50 m or off-screen = broken calibration
    print("TTA clips", len(todo), todo, flush=True)
    yolo = YOLO(ROOT + r"\Q1-Q3work\yolo11x.pt"); out = {}
    for c in todo:
        g = geo(c); fr = read_frames(c, list(g["d"]["idx"])); out[c] = []
        for f in fr:
            r = yolo.predict(f, imgsz=1280, conf=0.05, classes=[0], augment=True, verbose=False, device=0)[0]; out[c].append((r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()))
    pickle.dump(out, open(f"{PC}/kid_tta.pkl", "wb")); print("done", flush=True)

elif stage == "tta_pick":
    R = pickle.load(open(f"{PC}/kid_combo.pkl", "rb")); base = R["G* + re-pick nearest (>6 m)"]; TT = pickle.load(open(f"{PC}/kid_tta.pkl", "rb")); out = dict(base); kw = dict(vmax=2, slack=0.6, damp=0.0)
    for c, ex in TT.items():
        g2 = augment(geo(c), ex); t = base[c]; P_ = g2["P"][-1]; ds = np.array([spot_m(g2, b) for b in P_]); io = iou_many(t["kick"][-1], P_)
        cand = np.where((ds < 2) & (io < 0.3))[0]
        print(c, "TTA persons at the last frame", len(P_), "vs det", len(geo(c)["P"][-1]), "| nearest to the spot", f"{np.nanmin(ds):.2f} m", "| candidates", len(cand))
        if not len(cand): continue
        i = cand[ds[cand].argmin()]; kb, m = gtrack(g2, len(P_) and len(g2["P"]) - 1, P_[i], **kw)
        out[c] = dict(t, kick=kb, kick_a=kb[ANCHOR]); print("   re-picked: frames matched", int(m.sum()), "| new spot distance", f"{spot_m(g2, kb[-1]):.2f}", "| approach v", f"{approach(g2, kb)['v']:.2f}", "vs old", f"{approach(g2, t['kick'])['v']:.2f}")
    evaluate(out, name="G* + re-pick + TTA kicker"); R["G* + re-pick + TTA kicker"] = out; pickle.dump(R, open(f"{PC}/kid_combo.pkl", "wb"))

elif stage == "v2":                                                        # the motion cue inside the tracker (mono gate) and in the re-pick
    from ultralytics import YOLO
    TT = pickle.load(open(f"{PC}/kid_tta.pkl", "rb")); yolo = [None]

    def tta(c):
        if c not in TT:
            if yolo[0] is None: yolo[0] = YOLO(ROOT + r"\Q1-Q3work\yolo11x.pt")
            g = geo(c); TT[c] = [(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()) for r in
                                 (yolo[0].predict(f, imgsz=1280, conf=0.05, classes=[0], augment=True, verbose=False, device=0)[0] for f in read_frames(c, list(g["d"]["idx"])))]
        return augment(geo(c), TT[c])

    def base(mono):
        G = {}
        for c in CLIPS:
            g = geo(c)
            if g is None or T0.get(c) is None: continue
            a = np.asarray(T0[c]["kick_a"], float); kb, m = gtrack(g, ANCHOR, a, tgt=target(g, a[3] - a[1]), mono=mono); G[c] = dict(T0[c], kick=kb, kick_a=a)
        return G

    def repick(G, how, far=6.0, near=2.0, wide=8.0, vmin=1.5, mono=0.35):
        out, n = dict(G), []
        for c, t in G.items():
            g = geo(c)
            if g is None or g["gr"] is None or not calib_ok(g): continue
            s = spot_m(g, t["kick"][-1])
            if not (far < s < 50): continue
            g2 = tta(c); P_ = g2["P"][-1]
            if not len(P_): continue
            ds = np.array([spot_m(g2, b) for b in P_]); io = iou_many(t["kick"][-1], P_); new = None
            if how == "motion":                                            # the user's cue: the person near the spot who approaches the ball fastest
                cand = np.where((ds < wide) & (io < 0.3))[0]; best = (-np.inf, None)
                v_cur = approach(g2, t["kick"], use=LATE)["v"]
                for i in cand:
                    kb, m = gtrack(g2, len(g2["P"]) - 1, P_[i], tgt=target(g2, P_[i][3] - P_[i][1]), mono=mono); v = approach(g2, kb, use=LATE)["v"]
                    if np.isfinite(v) and v > best[0]: best = (v, kb)
                if best[1] is not None and best[0] >= vmin and best[0] >= (v_cur if np.isfinite(v_cur) else 0) + 1: new = best[1]
            if new is None:                                                # nearest person on the spot (kfix rule)
                cand = np.where((ds < near) & (io < 0.3))[0]
                if len(cand): i = cand[ds[cand].argmin()]; new = gtrack(g2, len(g2["P"]) - 1, P_[i], tgt=target(g2, P_[i][3] - P_[i][1]), mono=mono)[0]
            if new is not None: out[c] = dict(t, kick=new, kick_a=new[ANCHOR]); n.append(c)
        return out, n

    R = {}; LATE = slice(4, 12)                                            # approach over the last 0.4 s before the plant: a track that switched late shows it
    for mono in (None, 0.2, 0.35):
        G = base(mono); evaluate(G, name=f"G* mono {mono}")
        for how in ("nearest", "motion"):
            nm = f"G* mono {mono} + re-pick {how}"; R[nm], n = repick(G, how, mono=mono if mono is not None else 0.35); evaluate(R[nm], name=nm)
            print(f"   re-picked {len(n)}: {sorted(n)}", flush=True)
    pickle.dump(TT, open(f"{PC}/kid_tta.pkl", "wb")); old = pickle.load(open(f"{PC}/kid_combo.pkl", "rb")); old.update(R); pickle.dump(old, open(f"{PC}/kid_combo.pkl", "wb"))

elif stage == "sweep":                                                     # gate parameters of the tracker
    def iou(a, b):
        ix = np.maximum(0, np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0])); iy = np.maximum(0, np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1]))
        i = ix * iy; return i / np.maximum((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - i, 1e-6)
    grid = [(5, 0.3, 0.7), (3, 0.6, 0.7), (3, 0.75, 0.7), (2, 0.75, 0.7), (3, 0.75, 0.0), (3, 0.9, 0.7), (1.5, 0.9, 0.7), (5, 0.75, 0.7)] if len(sys.argv) < 3 else \
           [(3, 0.6, 0.0), (2, 0.6, 0.0), (4, 0.6, 0.0), (2, 0.75, 0.0), (4, 0.75, 0.0), (3, 0.9, 0.0), (3, 0.75, 0.3), (3, 0.5, 0.0)]
    for vmax, slack, damp in grid:
        G = {}
        for c in CLIPS:
            g = geo(c)
            if g is None or T0.get(c) is None: continue
            a = np.asarray(T0[c]["kick_a"], float); kb, m = gtrack(g, ANCHOR, a, vmax=vmax, slack=slack, damp=damp); G[c] = dict(T0[c], kick=kb, kick_a=a)
        ch = sum(np.nanmin(iou(np.asarray(T0[c]["kick"], float), G[c]["kick"])) < 0.5 for c in G)
        evaluate(G, name=f"G vmax {vmax} slack {slack} damp {damp}"); print(f"   clips changed vs T0 (some frame IoU < 0.5): {ch}", flush=True)
        pickle.dump(G, open(f"{PC}/kid_G_{vmax}_{slack}_{damp}.pkl", "wb"))
