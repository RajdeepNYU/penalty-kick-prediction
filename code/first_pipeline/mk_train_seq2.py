"""Derives train_seq2.py from train_seq.py: adds KEEPCH (keeper channels appended), AUX (auxiliary keeper-direction head) and JIT (frame dropout) options."""
p = r"C:\path\to\project-root\Q4_work"; s = open(f"{p}\\train_seq.py", encoding="utf-8").read()
def rep(a, b):
    global s; assert a in s, a[:70]; s = s.replace(a, b, 1)
rep('dev = "cuda"; Nc = X.shape[2]; ST = S.shape[1]', '''KEEPCH = int(os.environ.get("KEEPCH", 0)); AUX = int(os.environ.get("AUX", 0)); JIT = int(os.environ.get("JIT", 0))
Zk = (np.zeros((12, 49), np.float32), np.zeros(4, np.float32))
def _kd(R): return np.stack([R.get(c, (Zk, Zk))[0][0] for c in gt["clip"]]), np.stack([R.get(c, (Zk, Zk))[1][0] for c in gt["clip"]])
if KEEPCH:                                                    # append the keeper's per-frame channels to the kicker's
    RK = pickle.load(open(f"{C}/seqk.pkl", "rb")); XK, XKm = _kd(RK); X = np.concatenate([X, XK], 2); Xm = np.concatenate([Xm, XKm], 2)
    if AUGB:
        RKB = pickle.load(open(f"{C}/seqkB.pkl", "rb")) if os.path.exists(f"{C}/seqkB.pkl") else RK; XKB, XKBm = _kd(RKB); XB = np.concatenate([XB, XKB], 2); XBm = np.concatenate([XBm, XKBm], 2)
labg = {"left": 0, "stands in the middle": 1, "right": 2}; yg = gt["goalkeeper_direction"].map(labg).values; ygm = np.where(yg == 1, 1, 2 - yg)
dev = "cuda"; Nc = X.shape[2]; ST = S.shape[1]''')
rep('        s.head = nn.Sequential(nn.Linear(3 * H + ST, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 3))', '        s.head = nn.Sequential(nn.Linear(3 * H + ST, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 3)); s.aux = nn.Sequential(nn.Linear(3 * H + ST, 64), nn.GELU(), nn.Linear(64, 3))')
rep('    def forward(s, x, st):                                   # x [B,T,C]', '    def forward(s, x, st, aux=False):                        # x [B,T,C]')
rep('        return s.head(torch.cat([h[:, -1], h[:, -6:].mean(1), h.max(1).values, st], 1))', '        z = torch.cat([h[:, -1], h[:, -6:].mean(1), h.max(1).values, st], 1)\n        return (s.head(z), s.aux(z)) if aux else s.head(z)')
rep('    ty = torch.tensor(np.concatenate([y[tr]] * len(srcs)), device=dev); tym', '    tyg = torch.tensor(np.concatenate([yg[tr]] * len(srcs)), device=dev); tygm = torch.tensor(np.concatenate([ygm[tr]] * len(srcs)), device=dev)\n    ty = torch.tensor(np.concatenate([y[tr]] * len(srcs)), device=dev); tym')
rep('            xb = xb + 0.03 * torch.randn_like(xb) * (xb != 0)\n            loss = F.cross_entropy(m(xb, sb), yb, label_smoothing=0.1);',
    '            xb = xb + 0.03 * torch.randn_like(xb) * (xb != 0)\n            if JIT: xb = xb * (torch.rand(xb.shape[0], xb.shape[1], 1, device=dev) > 0.12).float()\n            if AUX:\n                ygb = torch.where(fl, tygm[ix], tyg[ix]); lo, la = m(xb, sb, aux=True); loss = F.cross_entropy(lo, yb, label_smoothing=0.1) + 0.5 * F.cross_entropy(la, ygb, label_smoothing=0.1)\n            else: loss = F.cross_entropy(m(xb, sb), yb, label_smoothing=0.1)\n           ')
rep('np.save(f"{C}/oof_seq_{EP}_{H}_k{KIN}_s{SPF}_{ARCH}_b{AUGB}.npy", P)', 'np.save(f"{C}/oof_seq2_{EP}_{H}_b{AUGB}_kc{KEEPCH}_ax{AUX}_j{JIT}.npy", P)')
rep('print(f"SEQ NET (epochs {EP}, hidden {H}): kick acc', 'print(f"SEQ NET (epochs {EP}, hidden {H}, keeper-ch {KEEPCH}, aux {AUX}, jitter {JIT}): kick acc')
open(f"{p}\\train_seq2.py", "w", encoding="utf-8").write(s); print("wrote train_seq2.py")
