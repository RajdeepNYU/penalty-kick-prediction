"""Stage 14: keeper-direction CNN trained on ALL public clips (ResNet34, goal-aligned keeper crops, flip aug), several seeds,
predicting the private clips with flip TTA. Saves cache/cnn_gk_private.npy  [900,3] in order (left, stands in the middle, right)."""
import os  # notebook: paths from the environment
import os, pickle, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, timm
os.environ["HF_HUB_OFFLINE"] = "1"
ROOT = os.environ.get("Q4_ROOT", r"C:\path\to\project-root")
D = ROOT + r"\Q4data"; C = os.environ["Q4_CACHE"]
ARCH, EP, SEEDS, GIDX = "resnet34.a1_in1k", 30, [0, 1, 2, 3, 4], [0, 3]
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv")
classes = ["left", "stands in the middle", "right"]
y_tr = gt["goalkeeper_direction"].map({c: i for i, c in enumerate(classes)}).values
foot_tr = np.where(gt["preferred_foot"] == "right", 1.0, -1.0).astype(np.float32)
foot_pv = np.where(pm["preferred_foot"] == "right", 1.0, -1.0).astype(np.float32)


def load(clips):
    X = np.zeros((len(clips), len(GIDX), 224, 224, 3), np.uint8)
    for i, c in enumerate(clips):
        a = pickle.load(open(f"{C}/align/{c}.pkl", "rb"))
        if a is not None: X[i] = a["keep"][GIDX]
    return X


Xtr, Xpv = load(gt["clip"]), load(pm["clip"])
mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 1, 3, 1, 1) * 255
std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 1, 3, 1, 1) * 255


class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.b = timm.create_model(ARCH, pretrained=True, num_classes=0)
        s.h = nn.Sequential(nn.Dropout(0.3), nn.Linear(s.b.num_features * len(GIDX) + 1, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 3))

    def forward(s, x, foot):
        B = x.shape[0]; return s.h(torch.cat([s.b(x.flatten(0, 1)).view(B, -1), foot[:, None]], 1))


def batch(X, foot_a, y_a, idx, flip, train):
    x = torch.from_numpy(X[idx]).cuda().permute(0, 1, 4, 2, 3).float(); foot = torch.from_numpy(foot_a[idx]).cuda()
    y = torch.from_numpy(y_a[idx]).cuda() if y_a is not None else None
    fl = torch.from_numpy(flip).cuda()
    x = torch.where(fl.view(-1, 1, 1, 1, 1), x.flip(-1), x); foot = torch.where(fl, -foot, foot)
    if y is not None: y = torch.where(fl & (y != 1), 2 - y, y)
    if train:
        B = x.shape[0]
        x = x * (1 + 0.2 * (torch.rand(B, 1, 1, 1, 1, device="cuda") - 0.5)) + 20 * (torch.rand(B, 1, 1, 1, 1, device="cuda") - 0.5)
        s = np.random.uniform(0.85, 1.0); h = int(224 * s); ox = np.random.randint(0, 224 - h + 1); oy = np.random.randint(0, 224 - h + 1)
        x = F.interpolate(x[..., oy:oy + h, ox:ox + h].flatten(0, 1), size=224, mode="bilinear", align_corners=False).view(B, -1, 3, 224, 224)
    return (x - mean) / std, foot, y


@torch.no_grad()
def predict(model, X, foot_a):
    model.eval(); P = []
    for i in range(0, len(X), 32):
        ii = np.arange(i, min(i + 32, len(X))); n = len(ii)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x, f, _ = batch(X, foot_a, None, ii, np.zeros(n, bool), False); p0 = F.softmax(model(x, f).float(), 1)
            x, f, _ = batch(X, foot_a, None, ii, np.ones(n, bool), False); p1 = F.softmax(model(x, f).float(), 1)[:, [2, 1, 0]]
        P.append(((p0 + p1) / 2).cpu().numpy())
    return np.concatenate(P)


out = 0
for seed in SEEDS:
    torch.manual_seed(seed); np.random.seed(seed)
    model = Net().cuda().to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW([{"params": model.b.parameters(), "lr": 1e-4}, {"params": model.h.parameters(), "lr": 5e-4}], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-4, 5e-4], total_steps=EP * (len(Xtr) // 32), pct_start=0.15)
    for ep in range(EP):
        model.train(); perm = np.random.permutation(len(Xtr))
        for i in range(0, len(perm) - 31, 32):
            ii = perm[i:i + 32]; fl = np.random.rand(len(ii)) < 0.5
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x, f, y = batch(Xtr, foot_tr, y_tr, ii, fl, True); loss = F.cross_entropy(model(x, f).float(), y, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    out = out + predict(model, Xpv, foot_pv); print("seed", seed, "done, last loss", round(loss.item(), 3), flush=True)
np.save(f"{C}/cnn_gk_private.npy", out / len(SEEDS)); print("saved", (out / len(SEEDS)).shape)
