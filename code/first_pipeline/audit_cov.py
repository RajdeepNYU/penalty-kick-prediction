import pickle, numpy as np, pandas as pd
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv"); tracks = pickle.load(open(f"{C}/tracks.pkl", "rb"))
LEG = [11, 12, 13, 14, 15, 16]; FOOT = [17, 19, 20, 22]; BODY = [5, 6, 11, 12, 13, 14, 15, 16]
rows = []
for part, df in (("train", gt), ("private", pm)):
    for c in df["clip"]:
        t = tracks[c]; b = pickle.load(open(f"{C}/ballspot/{c}.pkl", "rb")); k = pickle.load(open(f"{C}/kpts/{c}.pkl", "rb")); kk = pickle.load(open(f"{C}/kkeep/{c}.pkl", "rb"))
        if t is None or b is None or k is None or kk is None or t["keep"] is None: rows.append(dict(part=part, clip=c, ok=False)); continue
        nF = b["nF"]; last8 = slice(max(0, nF - 8), nF)
        kv = np.isfinite(t["kick"][:, 0]).mean(); gv = np.isfinite(t["keep"][:, 0]).mean()
        kfr = (k["sc"][:, LEG] >= 2.0).all(1); kft = (k["sc"][:, FOOT] >= 2.0).mean(1); gfr = (kk["sc"][:, LEG] >= 2.0).all(1)
        bh = k["box"][:, 3] - k["box"][:, 1]; gh = kk["box"][:, 3] - kk["box"][:, 1]
        rows.append(dict(part=part, clip=c, ok=True, kick_track=kv, keep_track=gv, kick_legs_ok=kfr[last8].mean(), kick_foot_ok=kft[last8].mean(), keep_legs_ok=gfr[last8].mean(),
                         ball_support=b["support"], ball_none=not np.isfinite(b["ball_last"][0]), kbox_jump=np.nanmax(bh) / max(np.nanmin(bh), 1), gbox_jump=np.nanmax(gh) / max(np.nanmin(gh), 1), kh=np.median(bh), gh=np.median(gh)))
r = pd.DataFrame(rows); r.to_pickle(f"{C}/audit_cov.pkl"); a = r[r.ok]
print(f"clips {len(r)} (usable {len(a)}); unusable (no track/ball/keeper): {(~r.ok).sum()}")
for part in ("train", "private"):
    s = a[a.part == part]
    print(f"\n[{part}] n={len(s)}")
    print(f"  kicker tracked in all 13 timed frames: {(s.kick_track == 1).mean():.3f} | keeper: {(s.keep_track == 1).mean():.3f} (mean frames {13 * s.kick_track.mean():.1f}/13, {13 * s.keep_track.mean():.1f}/13)")
    print(f"  kicker: both legs (hip/knee/ankle x2) confident in >=90% of last 8 frames: {(s.kick_legs_ok >= .9).mean():.3f} | all 4 foot points (toe/heel x2) >=90%: {(s.kick_foot_ok >= .9).mean():.3f}")
    print(f"  keeper: both legs confident in >=90% of last 8 frames: {(s.keep_legs_ok >= .9).mean():.3f}")
    print(f"  ball: supported by >=3 frames {(s.ball_support >= 3).mean():.3f} | >=5 {(s.ball_support >= 5).mean():.3f} | none found {s.ball_none.mean():.3f}")
    print(f"  kicker box height jumps >1.4x within clip: {(s.kbox_jump > 1.4).mean():.3f} | keeper: {(s.gbox_jump > 1.4).mean():.3f} ; median heights px: kicker {s.kh.median():.0f}, keeper {s.gh.median():.0f}")
    allok = (s.kick_legs_ok >= .9) & (s.keep_legs_ok >= .9) & (s.ball_support >= 3) & (s.kick_track == 1) & (s.keep_track == 1)
    print(f"  ALL of kicker legs + keeper legs + ball(>=3) + full tracks: {allok.mean():.3f}")
print("\ngoalposts / goal frame detected: NOT DETECTED (0%): goal direction = ball -> keeper-feet proxy")
