"""Stage G1: goalposts and crossbar on the last frame (stabilised coordinates == last-frame coordinates).
LSD segments inside the GroundingDINO goal box -> white near-vertical fragments merged into posts -> best post pair (long, tops joined by a white crossbar,
keeper between them). Output per clip: post tops/bases, quality numbers. Writes cache/goalposts.pkl.
usage: python goalposts.py [montage.jpg N]"""
import sys, pickle, numpy as np, cv2
from multiprocessing import Pool
from common import *


def white_frac(hsv, p, q, vmin):
    n = max(int(np.hypot(*(q - p))), 2); t = np.linspace(0, 1, n); H, W = hsv.shape[:2]
    xs = np.clip((p[0] + t * (q[0] - p[0])).astype(int), 0, W - 1); ys = np.clip((p[1] + t * (q[1] - p[1])).astype(int), 0, H - 1)
    best = 0.0
    for dy in (-2, -1, 0, 1, 2):                                   # tolerate small offsets of the bar
        yy = np.clip(ys + dy, 0, H - 1); best = max(best, float(((hsv[yy, xs, 2] >= vmin) & (hsv[yy, xs, 1] < 90)).mean()))
    return best


def detect(im, g, kfeet):
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    H, W = im.shape[:2]; x0, y0, x1, y1 = g[:4]; gw, gh = x1 - x0, y1 - y0; pad = 0.15
    X0, Y0, X1, Y1 = int(max(x0 - pad * gw, 0)), int(max(y0 - pad * gh, 0)), int(min(x1 + pad * gw, W)), int(min(y1 + pad * gh, H))
    crop = im[Y0:Y1, X0:X1]
    if crop.shape[0] < 10 or crop.shape[1] < 10: return None
    sc = max(1.0, 500 / max(crop.shape[:2])); up = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
    L = lsd.detect(cv2.cvtColor(up, cv2.COLOR_BGR2GRAY))[0]
    if L is None: return None
    L = L.reshape(-1, 4) / sc + np.array([X0, Y0, X0, Y0]); hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    vref = np.percentile(hsv[Y0:Y1, X0:X1, 2], 97); vmin = max(120.0, 0.75 * vref)
    seg = []
    for xa, ya, xb, yb in L:
        if ya > yb: xa, ya, xb, yb = xb, yb, xa, ya
        ln = np.hypot(xb - xa, yb - ya); ang = np.degrees(np.arctan2(yb - ya, xb - xa))
        if ln < 0.08 * gh or abs(ang - 90) > 22: continue
        if white_frac(hsv, np.array([xa, ya]), np.array([xb, yb]), vmin) < 0.6: continue
        seg.append([xa, ya, xb, yb, ln])
    if len(seg) < 2: return dict(ok=0, nseg=len(seg))
    seg = np.array(seg)
    # merge collinear fragments: line x = a*y + b fitted per segment; group if x at the other segment's mid-y within tol
    tol = max(3.0, 0.012 * gw); groups = []
    for s in seg[np.argsort(-seg[:, 4])]:
        a = (s[2] - s[0]) / max(s[3] - s[1], 1e-6); b = s[0] - a * s[1]; placed = False
        for G in groups:
            ym = (s[1] + s[3]) / 2; xg = G["a"] * ym + G["b"]
            if abs(xg - (a * ym + b)) < tol and abs(G["a"] - a) < 0.15: G["s"].append(s); placed = True; break
        if not placed: groups.append(dict(a=a, b=b, s=[s]))
    posts = []
    for G in groups:
        S = np.array(G["s"]); pts = np.r_[S[:, :2], S[:, 2:4]]; A_ = np.c_[pts[:, 1], np.ones(len(pts))]; a, b = np.linalg.lstsq(A_, pts[:, 0], rcond=None)[0]
        yt, yb = pts[:, 1].min(), pts[:, 1].max(); cover = S[:, 4].sum() / max(yb - yt, 1)
        posts.append(dict(top=np.array([a * yt + b, yt]), base=np.array([a * yb + b, yb]), len=yb - yt, cover=min(cover, 1.5), a=a))
    posts = [p for p in posts if p["len"] > 0.25 * gh]
    if len(posts) < 2: return dict(ok=0, nseg=len(seg), nposts=len(posts))
    best, bs = None, -1
    for i in range(len(posts)):
        for j in range(i + 1, len(posts)):
            p, q = posts[i], posts[j]; sep = abs(p["top"][0] - q["top"][0])
            if sep < 0.12 * gw: continue
            bar = white_frac(hsv, p["top"], q["top"], vmin)
            lo, hi = sorted([p["top"][0], q["top"][0]]); between = float(lo - 0.1 * gw <= kfeet[0] <= hi + 0.1 * gw)
            score = (p["len"] + q["len"]) / gh + 1.5 * bar + 0.7 * between + 0.3 * sep / gw
            if score > bs: bs, best = score, (p, q, bar, between)
    if best is None: return dict(ok=0, nseg=len(seg), nposts=len(posts))
    p, q, bar, between = best
    return dict(ok=1, top=np.array([p["top"], q["top"]]), base=np.array([p["base"], q["base"]]), plen=np.array([p["len"], q["len"]]), bar=bar, between=between, score=bs, nposts=len(posts), vmin=vmin)


def work(c):
    b = load("ballspot", c); s = load("scene", c); t = tracks()[c]
    if b is None or s is None or t is None or t["keep"] is None or not len(s["goal_l"]): return c, None
    kb = t["keep"]; vv = np.where(np.isfinite(kb[:, 0]))[0]; k = kb[vv[-1]]; kfeet = np.array([(k[0] + k[2]) / 2, k[3]])
    im = read_frames(c, [b["F"][-1]])[0]; g = s["goal_l"][0]
    r = detect(im, g, kfeet)
    if r is not None: r["goal"] = g; r["kfeet"] = kfeet
    return c, r


if __name__ == "__main__":
    with Pool(int(os.environ.get("NPOOL", "12"))) as pool: res = dict(pool.map(work, CLIPS, chunksize=8))
    pickle.dump(res, open(f"{PC}/goalposts.pkl", "wb"))
    ok = np.array([r is not None and r.get("ok", 0) == 1 for r in res.values()])
    print(f"posts found: {ok.mean():.3f} ({ok.sum()} of {len(ok)}) | crossbar>0.5: {np.mean([r['bar'] > 0.5 for r in res.values() if r and r.get('ok')]):.3f} | keeper between: {np.mean([r['between'] for r in res.values() if r and r.get('ok')]):.3f}")
    if len(sys.argv) > 2:
        out, n = sys.argv[1], int(sys.argv[2]); rng = np.random.RandomState(2); cl = list(rng.choice(CLIPS, n, replace=False)); tiles = []
        for c in cl:
            r = res[c]; b = load("ballspot", c); f = read_frames(c, [b["F"][-1]])[0].copy()
            if r is None: continue
            g = r["goal"]; cv2.rectangle(f, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), (255, 0, 255), 1)
            if r.get("ok"):
                for i in range(2): cv2.line(f, tuple(r["top"][i].astype(int)), tuple(r["base"][i].astype(int)), (0, 255, 255), 2); cv2.circle(f, tuple(r["base"][i].astype(int)), 5, (0, 0, 255), -1)
                cv2.line(f, tuple(r["top"][0].astype(int)), tuple(r["top"][1].astype(int)), (0, 128, 255), 1)
            x0, y0, x1, y1 = g[:4]; w, h = x1 - x0, y1 - y0
            crop = f[int(max(y0 - 0.3 * h, 0)):int(min(y1 + 0.3 * h, f.shape[0])), int(max(x0 - 0.3 * w, 0)):int(min(x1 + 0.3 * w, f.shape[1]))]
            crop = cv2.resize(crop, (480, 300)); cv2.putText(crop, f"{c} ok={r.get('ok')} bar={r.get('bar', 0):.2f}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2); tiles.append(crop)
        while len(tiles) % 4: tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
