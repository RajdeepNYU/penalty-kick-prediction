"""E1: the user's kin2 kicker features (Q4_work/kin2.build, unchanged) recomputed with a different stationary ball. usage: python kin2b.py STREAM BALL
STREAM in A / B / E / F / G (streams.py); BALL in old / ball2.  -> cache/kin2_{STREAM}_{BALL}.pkl"""
import sys, pickle, warnings, numpy as np
from multiprocessing import Pool
warnings.filterwarnings("ignore")
from common import *
import kin2

STREAM, BALL = sys.argv[1], sys.argv[2]
B2 = pickle.load(open(f"{PC}/{BALL}.pkl", "rb")) if BALL.startswith("ball") else {}


def one(c):
    from streams import stream_kb
    b = load("ballspot", c); kb = stream_kb(c, STREAM); t = tracks()[c]
    if b is None or kb is None or t is None or t["keep"] is None: return c, None, None
    if B2.get(c) is not None: b = dict(b); b["ball_last"] = np.asarray(B2[c]["ball"], float)
    kin2.NAMES.clear(); a = kin2.build(b, kb, t, META[c], False); names = list(kin2.NAMES); kin2.NAMES.clear()
    m = kin2.build(b, kb, t, META[c], True)
    return c, (a, m) if a is not None and m is not None else None, names


if __name__ == "__main__":
    with Pool(int(os.environ.get("NPOOL", "12"))) as p: res = p.map(one, CLIPS, chunksize=20)
    rows = {c: r for c, r, _ in res if r is not None}; names = next(n for _, r, n in res if r is not None)
    pickle.dump(dict(rows=rows, names=names), open(f"{PC}/kin2_{STREAM}_{BALL}.pkl", "wb")); print(STREAM, BALL, "built", len(rows), "dim", len(names))
