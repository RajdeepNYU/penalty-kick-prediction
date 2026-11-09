import pickle, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv"); tracks = pickle.load(open(f"{C}/tracks.pkl", "rb")); ball = pickle.load(open(f"{C}/ball_final.pkl", "rb"))
LEG = [11, 12, 13, 14, 15, 16]; FOOT = [15, 16, 17, 19, 20, 22]; rows = []
for part, df in (("train", gt), ("private", pm)):
    for c in df["clip"]:
        t = tracks[c]; b = pickle.load(open(f"{C}/ballspot/{c}.pkl", "rb")); k = pickle.load(open(f"{C}/kpts/{c}.pkl", "rb")); kk = pickle.load(open(f"{C}/kkeep/{c}.pkl", "rb")); s = pickle.load(open(f"{C}/scene/{c}.pkl", "rb"))
        l8 = slice(max(0, b["nF"] - 8), b["nF"]); legs = np.median(k["sc"][l8][:, LEG].min(1)); foot = np.median(k["sc"][l8][:, FOOT].min(1)); kleg = np.median(kk["sc"][l8][:, LEG].min(1))
        br = ball[c]; ball_ok = (br["method"] == "consensus" and br["support"] >= 3) or (br["method"] != "consensus" and br["d_feet"] < 0.6)
        rows.append(dict(part=part, clip=c, kick_track=np.isfinite(t["kick"][:, 0]).mean(), keep_track=np.isfinite(t["keep"][:, 0]).mean(), kick_legs=legs >= 2.0, kick_feet=foot >= 2.0, keep_legs=kleg >= 2.0,
                         ball_ok=bool(ball_ok), ball_strong=br["support"] >= 3, goal=len(s["goal_a"]) > 0 and s["goal_a"][0][4] >= 0.3, goal_any=len(s["goal_a"]) > 0, kh=k["box"][-1][3] - k["box"][-1][1]))
r = pd.DataFrame(rows); r["healthy"] = (r.kick_track >= 12 / 13) & r.kick_legs & r.ball_ok & r.goal & (r.keep_track >= 12 / 13); r.to_pickle(f"{C}/audit_final.pkl")
for part in ("train", "private"):
    s_ = r[r.part == part]; print(f"[{part}] n={len(s_)} | kicker tracked>=12/13 frames {(s_.kick_track >= 12 / 13).mean():.3f} | kicker legs (median conf) {s_.kick_legs.mean():.3f} | kicker feet kp {s_.kick_feet.mean():.3f} | ball ok (fused) {s_.ball_ok.mean():.3f} (consensus support>=3: {s_.ball_strong.mean():.3f}) | goal found (score>=0.3) {s_.goal.mean():.3f} (any {s_.goal_any.mean():.3f}) | keeper tracked {(s_.keep_track >= 12 / 13).mean():.3f} | keeper legs {s_.keep_legs.mean():.3f}")
    print(f"        ALL kick-relevant detections healthy (kicker track+legs, ball, goal, keeper track): {s_.healthy.mean():.3f}")
# metrics on train with the latest OOF predictions
tr = r[r.part == "train"].reset_index(drop=True); Pk = np.load(f"{C}/new_Pk.npy"); Pg = np.load(f"{C}/new_Pg.npy"); yk, yg, yo = gt["kick_direction"].values, gt["goalkeeper_direction"].values, gt["goal_outcome"].values
okk = np.array(["left", "middle", "right"])[Pk.argmax(1)] == yk; okg = np.array(["left", "right", "stands in the middle"])[Pg.argmax(1)] == yg; goal = yo == "goal"; lr = yk != "middle"
def row(name, m): k, g, o = okk[m].mean(), okg[m].mean(), goal[m].mean(); print(f"  {name:30s} n={m.sum():4d} ({m.mean():4.0%}) | kick {k:.3f} (L/R-only {np.mean(np.where(lr, np.array(['left', 'middle', 'right'])[Pk.argmax(1)] == yk, np.nan)[m][lr[m]]):.3f}) | keeper {g:.3f} | outcome {o:.3f} | SCORE {0.6 * k + 0.25 * g + 0.15 * o:.3f}")
print("\nout-of-fold metrics, healthy vs not (train clips, current pipeline):"); h = tr.healthy.values; row("ALL", np.ones(len(tr), bool)); row("HEALTHY", h); row("NOT healthy", ~h)
for nm, m in (("ball not ok", ~tr.ball_ok.values), ("kicker legs weak", ~tr.kick_legs.values), ("kicker track < 12/13", tr.kick_track.values < 12 / 13), ("keeper track < 12/13", tr.keep_track.values < 12 / 13), ("goal not found", ~tr.goal.values)):
    if m.sum(): row(nm, m)
