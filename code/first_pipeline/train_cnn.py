"""Stage 9: fine-tune a pretrained CNN on goal-aligned crops (shared backbone over frames, late fusion), flip+label-swap aug.
usage: python train_cnn.py <arch> <target kick|gk> <epochs> <folds e.g. 0,1,2,3,4> [seed]"""
import os, sys, pickle, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, timm
from sklearn.model_selection import StratifiedGroupKFold
os.environ["HF_HUB_OFFLINE"] = "1"
ROOT = r"C:\path\to\project-root"
D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
arch, target, EP, folds = sys.argv[1], sys.argv[2], int(sys.argv[3]), [int(x) for x in sys.argv[4].split(",")]
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
KIDX = [0, 2, 4]          # kicker frames: 0, .06, .15 s (of align.py KF)
GIDX = [int(v) for v in os.environ.get("GIDX", "0,3").split(",")]   # keeper frames: 0, .25 s (align.py GF = 0, .06, .15, .25 s)
torch.manual_seed(seed); np.random.seed(seed)
gt = pd.read_csv(f"{D}/train_ground_truth.csv")
col = {"kick": "kick_direction", "gk": "goalkeeper_direction"}[target]
classes = ["left", "middle", "right"] if target == "kick" else ["left", "stands in the middle", "right"]
y_all = gt[col].map({c: i for i, c in enumerate(classes)}).values
foot_all = np.where(gt["preferred_foot"] == "right", 1.0, -1.0).astype(np.float32)
grp = gt["source_id"].values
key, IDX = ("kick", KIDX) if target == "kick" else ("keep", GIDX)
X = np.zeros((len(gt), len(IDX), 224, 224, 3), np.uint8)
for i, c in enumerate(gt["clip"]):
    a = pickle.load(open(f"{C}/align/{c}.pkl", "rb"))
    X[i] = a[key][IDX]
print("loaded", X.shape, flush=True)
mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 1, 3, 1, 1) * 255
std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 1, 3, 1, 1) * 255


class Net(nn.Module):
    def __init__(s, nf):
        super().__init__()
        s.b = timm.create_model(arch, pretrained=True, num_classes=0, drop_path_rate=0.1 if "convnext" in arch else None) if "convnext" in arch else timm.create_model(arch, pretrained=True, num_classes=0)
        d = s.b.num_features
        s.h = nn.Sequential(nn.Dropout(0.3), nn.Linear(d * nf + 1, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 3))

    def forward(s, x, foot):  # x [B,F,3,H,W]
        B, Fn = x.shape[:2]
        f = s.b(x.flatten(0, 1)).view(B, -1)
        return s.h(torch.cat([f, foot[:, None]], 1))


def batch(idx, flip, train):
    x = torch.from_numpy(X[idx]).cuda().permute(0, 1, 4, 2, 3).float()      # [B,F,3,H,W]
    foot = torch.from_numpy(foot_all[idx]).cuda(); y = torch.from_numpy(y_all[idx]).cuda()
    if flip is not None:
        fl = torch.from_numpy(flip).cuda()
        x = torch.where(fl.view(-1, 1, 1, 1, 1), x.flip(-1), x)
        foot = torch.where(fl, -foot, foot); y = torch.where(fl & (y != 1), 2 - y, y) if target == "kick" else torch.where(fl & (y != 1), 2 - y, y)
    if train:
        B = x.shape[0]
        x = x * (1 + 0.2 * (torch.rand(B, 1, 1, 1, 1, device="cuda") - 0.5)) + 20 * (torch.rand(B, 1, 1, 1, 1, device="cuda") - 0.5)
        s = np.random.uniform(0.85, 1.0); ox = np.random.randint(0, int(224 * (1 - s)) + 1); oy = np.random.randint(0, int(224 * (1 - s)) + 1)
        h = int(224 * s); x = x[..., oy:oy + h, ox:ox + h]
        x = F.interpolate(x.flatten(0, 1), size=224, mode="bilinear", align_corners=False).view(B, -1, 3, 224, 224)
    return (x - mean) / std, foot, y


@torch.no_grad()
def predict(model, idx):
    model.eval(); P = []
    for i in range(0, len(idx), 32):
        ii = idx[i:i + 32]; n = len(ii)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x, f, _ = batch(ii, np.zeros(n, bool), False); p0 = F.softmax(model(x, f).float(), 1)
            x, f, _ = batch(ii, np.ones(n, bool), False); p1 = F.softmax(model(x, f).float(), 1)
        p1 = p1[:, [2, 1, 0]]    # mirrored prediction swaps left/right
        P.append(((p0 + p1) / 2).cpu().numpy())
    return np.concatenate(P)


splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y_all, grp))
curves = []; oof = np.zeros((len(gt), 3)); have = np.zeros(len(gt), bool)
for fo in folds:
    tr, te = splits[fo]
    model = Net(len(IDX)).cuda().to(memory_format=torch.channels_last)
    BLR = float(os.environ.get("BLR", 1e-4))
    opt = torch.optim.AdamW([{"params": model.b.parameters(), "lr": BLR}, {"params": model.h.parameters(), "lr": 5e-4}], weight_decay=0.05)
    steps = EP * (len(tr) // 32); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[BLR, 5e-4], total_steps=steps, pct_start=0.15)
    cur = []; t0 = time.time()
    for ep in range(EP):
        model.train(); perm = np.random.permutation(tr)
        for i in range(0, len(perm) - 31, 32):
            ii = perm[i:i + 32]; fl = np.random.rand(len(ii)) < 0.5
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x, f, y = batch(ii, fl, True); loss = F.cross_entropy(model(x, f).float(), y, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        p = predict(model, te); acc = (p.argmax(1) == y_all[te]).mean(); cur.append(acc)
        print(f"fold {fo} ep {ep + 1}/{EP} loss {loss.item():.3f} val acc {acc:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    oof[te] = p; have[te] = True; curves.append(cur)
curves = np.array(curves)
print("mean val acc by epoch:", np.round(curves.mean(0), 3))
print(f"final-epoch acc over folds {sorted(folds)}: {curves[:, -1].mean():.3f} | pooled {(oof[have].argmax(1) == y_all[have]).mean():.3f}")
np.save(f"{C}/oof_{target}_{arch}_{seed}.npy", np.where(have[:, None], oof, np.nan))
