"""Temporal pose network of Q4_work/train_seq2.py (same architecture, optimiser, mirror aug + TTA, extra measurement streams as extra samples),
reading the sequence caches rebuilt here (seq2b.py). Default config = the v10 bag member (AUGB=1, no keeper channels / aux / jitter).
usage: [SEQ=b2|b2R] [STREAMS=EF|G] [MODE=cv|full] python train_seqP.py EPOCHS HIDDEN
  cv   -> cache/oofP_{EP}_{H}_st{STREAMS}_{SEQ}.npy   (source-grouped 5-fold x 3 partitions, OOF probs left/middle/right)
  full -> cache/prvP_{EP}_{H}_st{STREAMS}_{SEQ}.npy   (3 seeds on all public clips, private-clip probs in pm order)"""
import os, pickle, sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from common import gt, pm, PC

EP = int(sys.argv[1]) if len(sys.argv) > 1 else 12; H = int(sys.argv[2]) if len(sys.argv) > 2 else 96
SEQ = os.environ.get("SEQ", "b2"); STREAMS = os.environ.get("STREAMS", ""); MODE = os.environ.get("MODE", "cv")
SEEDS = tuple(int(s) for s in os.environ.get("SEEDS", "0,1,2").split(","))                 # partition / init seeds; other than 0,1,2 -> tag suffix _sd...
TARGET = os.environ.get("TARGET", "kick"); BASE = os.environ.get("BASE", "AB")             # TARGET=keeper: goalkeeper_direction (left / stands / right), e.g. with BASE=AF SEQ=kp
COL = "goalkeeper_direction" if TARGET == "keeper" else "kick_direction"                 # TARGET=kicklr: kick, trained on the non-middle clips only (left vs right)
lab = {"left": 0, "middle": 1, "right": 2, "stands in the middle": 1}; y = gt[COL].map(lab).values; grp = gt["source_id"].values; ym = np.where(y == 1, 1, 2 - y)
DROPIN = float(os.environ.get("DROPIN", "0"))                                            # training-time input-channel dropout probability
SWT = os.environ.get("SWT", "")                                                        # 1: loss weight 1 / n(source), 2: 1 / sqrt(n(source))
nsrc = gt.groupby("source_id")["clip"].transform("count").values.astype(float); wsrc = {"": np.ones(len(gt)), "1": 1 / nsrc, "2": 1 / np.sqrt(nsrc)}[SWT]
TR = list(gt["clip"]); PR = list(pm["clip"]); ALL = TR + PR


def load_stream(L, sfx=""):
    R = pickle.load(open(f"{PC}/seq_{L}_{SEQ}{sfx}.pkl", "rb")); r0 = next(iter(R.values())); Z = (np.zeros_like(r0[0][0]), np.zeros_like(r0[0][1]))
    get = lambda c, j: R.get(c, (Z, Z))[j]
    return tuple(np.stack([get(c, j)[q] for c in ALL]) for j, q in ((0, 0), (1, 0), (0, 1), (1, 1))) + (np.array([c in R for c in ALL]),)     # X, Xm, S, Sm, ok


SRC = [load_stream(L) for L in BASE] + [load_stream(L) for L in STREAMS]
AUGS = [a for a in os.environ.get("AUGS", "").split(",") if a]                  # time-shifted windows (e.g. s1,s2) of every stream: extra training samples only
TRX = [load_stream(L, f"_{a}") for a in AUGS for L in list(BASE) + list(STREAMS)]
X0 = SRC[0][0]; Nc = X0.shape[2]; ST = SRC[0][2].shape[1]; dev = "cuda"; nT = len(TR)


ARCH = os.environ.get("ARCH", "conv")                                            # conv (v10) | gru: bidirectional GRU encoder, same input layer and head


class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.inp = nn.Sequential(nn.Linear(Nc, H), nn.GELU(), nn.Dropout(0.1))
        if ARCH == "gru": s.rnn = nn.GRU(H, H // 2, batch_first=True, bidirectional=True); s.rnorm = nn.LayerNorm(H); s.drop = nn.Dropout(0.2)
        else: s.convs = nn.ModuleList([nn.Conv1d(H, H, 3, padding=d, dilation=d) for d in (1, 2, 1)]); s.drop = nn.Dropout(0.2); s.norm = nn.ModuleList([nn.LayerNorm(H) for _ in range(3)])
        s.head = nn.Sequential(nn.Linear(3 * H + ST, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 3))

    def forward(s, x, st):
        h = s.inp(x)
        if ARCH == "gru": h = s.rnorm(h + s.drop(s.rnn(h)[0]))
        else:
            for conv, nm in zip(s.convs, s.norm): h = nm(h + s.drop(F.gelu(conv(h.transpose(1, 2)).transpose(1, 2))))
        return s.head(torch.cat([h[:, -1], h[:, -6:].mean(1), h.max(1).values, st], 1))


def run(tr, te, seed):
    """tr: indices into the public clips, te: indices into ALL"""
    torch.manual_seed(seed); np.random.seed(seed)
    if TARGET == "kicklr": tr = tr[y[tr] != 1]
    X, Xm, S, Sm, _ = SRC[0]
    both = np.concatenate([X[tr], Xm[tr]]); mu = both.reshape(-1, Nc).mean(0); sd = both.reshape(-1, Nc).std(0) + 1e-3; sd[-3:] = 1; mu[-3:] = 0
    nz = lambda a: np.clip((a - mu) / sd, -6, 6).astype(np.float32)
    sb_ = np.concatenate([S[tr], Sm[tr]]); smu = np.nanmean(sb_, 0); ssd = np.nanstd(sb_, 0) + 1e-3; smu[:4] = 0; ssd[:4] = 1
    nzs = lambda a: np.nan_to_num(np.clip((a - smu) / ssd, -6, 6), nan=0.0).astype(np.float32)
    T_ = lambda a: torch.tensor(a, device=dev)
    trs = [(a, tr) for a in SRC] + [(a, tr[a[4][tr]]) for a in TRX]
    tX = T_(np.concatenate([nz(a[0][i]) for a, i in trs])); tXm = T_(np.concatenate([nz(a[1][i]) for a, i in trs]))
    tS = T_(np.concatenate([nzs(a[2][i]) for a, i in trs])); tSm = T_(np.concatenate([nzs(a[3][i]) for a, i in trs]))
    ty = T_(np.concatenate([y[i] for _, i in trs])); tym = T_(np.concatenate([ym[i] for _, i in trs]))
    tw = np.concatenate([wsrc[i] for _, i in trs]); tw = T_((tw / tw.mean()).astype(np.float32))
    ev = [[T_(a) for a in (nz(s_[0][te]), nz(s_[1][te]), nzs(s_[2][te]), nzs(s_[3][te]))] for s_ in SRC]
    m = Net().to(dev); opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-2); n = len(tX); steps = EP * ((n + 63) // 64)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps, pct_start=0.15)
    for ep in range(EP):
        m.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, 64):
            ix = perm[i:i + 64]; fl = torch.rand(len(ix), device=dev) < 0.5
            xb = torch.where(fl[:, None, None], tXm[ix], tX[ix]); sb = torch.where(fl[:, None], tSm[ix], tS[ix]); yb = torch.where(fl, tym[ix], ty[ix])
            xb = xb + 0.03 * torch.randn_like(xb) * (xb != 0)
            if DROPIN: xb = xb * (torch.rand(len(ix), 1, Nc, device=dev) >= DROPIN)       # whole input channels masked per sample (robustness to noisy measurements)
            loss = F.cross_entropy(m(xb, sb), yb, label_smoothing=0.1) if not SWT else (F.cross_entropy(m(xb, sb), yb, label_smoothing=0.1, reduction="none") * tw[ix]).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
    m.eval()
    with torch.no_grad():
        outs = [(F.softmax(m(e[0], e[2]), 1) + F.softmax(m(e[1], e[3]), 1)[:, [2, 1, 0]]) / 2 for e in ev]
    return (sum(outs) / len(outs)).cpu().numpy()


t0 = time.time(); seeds = SEEDS; tag = f"{EP}_{H}_st{STREAMS}_{SEQ}" + (f"_aug{''.join(AUGS)}" if AUGS else "") + (f"_sd{''.join(map(str, SEEDS))}" if SEEDS != (0, 1, 2) else "") \
    + (f"_{TARGET}{BASE}" if TARGET != "kick" or BASE != "AB" else "") + (f"_{ARCH}" if ARCH != "conv" else "") + (f"_sw{SWT}" if SWT else "") + (f"_di{DROPIN:g}" if DROPIN else "")
if MODE == "cv":
    P = np.zeros((nT, 3))
    for s in seeds:
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(np.zeros(nT), gt[COL].values, grp): P[te] += run(tr, te, s * 10) / len(seeds)
    cl = np.array(["left", "middle", "right"] if TARGET != "keeper" else ["left", "stands in the middle", "right"]); pred = cl[P.argmax(1)]; yk = gt[COL].values; lr = yk != cl[1]
    print(f"SEQ-P {tag}: {TARGET} acc {(pred == yk).mean():.4f} | L/R-only {(pred[lr] == yk[lr]).mean():.4f}  [{time.time() - t0:.0f}s]", flush=True)
    np.save(f"{PC}/oofP_{tag}.npy", P)
else:
    Q = sum(run(np.arange(nT), np.arange(nT, len(ALL)), s * 10) for s in seeds) / len(seeds)
    np.save(f"{PC}/prvP_{tag}.npy", Q); print(f"SEQ-P full {tag}: private preds {Q.shape}  [{time.time() - t0:.0f}s]")
