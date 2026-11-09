"""Assemble Q4_word_Prime/Q4_Penalty_Kick_final.ipynb for the final recipe N6 (N4's kick + the G3 keeper), replacing the P3 notebook of 25 Oct
(build_nb_final.py and its notebooks are kept in archive/). Every script that produced the submission is written by a %%writefile cell into a clean-room
working folder nbrun/ (laid out like Q4_word_Prime: scripts at the top, cache/, cache_kf/, cache_g/, wcache/ = the Q4_work perception cache, logs/) and run
there as a subprocess with all paths from the environment, so a notebook run never writes into Q4_work or into Q4_word_Prime's own caches and CSVs.
Stages are either 'heavy' (GPU / video: detection, pose, dense pose, MotionBERT, calibration, crops, embeddings, networks; cached unless HEAVY=True)
or refit stages (feature tables, tree and keeper fits, the pixel classifier, every CSV step; recomputed when REFIT=True, the older output moved to nbrun/_prev
and compared). The last cells compare the CSV with submission_N6.csv. usage: python build_nb_N6.py"""
import os, re, ast, nbformat as nbf

R = r"C:\path\to\project-root"; Q = R + r"\Q4_work"; P = R + r"\Q4_word_Prime"
rd = lambda p: open(p, encoding="utf-8-sig").read()     # utf-8-sig: two scripts start with a byte-order mark


def after_doc(s, line):
    i = s.index('"""', 3) + 3 if s.startswith('"""') else 0
    return s[:i] + "\n" + line + s[i:]


def adaptQ(s, name):
    """the user's Q4_work stage scripts: cache dir and project root from the environment (no write can reach Q4_work/cache)"""
    s = s.replace(r'ROOT = r"C:\path\to\project-root"', r'ROOT = os.environ.get("Q4_ROOT", r"C:\path\to\project-root")')
    s = s.replace(r'C = ROOT + r"\Q4_work\cache"', r'C = os.environ["Q4_CACHE"]').replace(r'OUT = ROOT + r"\Q4_work\cache\det"', r'OUT = os.environ["Q4_CACHE"] + r"\det"')
    assert "Q4_work" not in s.replace("Q4_work/kin2", "").replace("Q4_work/", ""), (name, [l for l in s.split("\n") if "Q4_work" in l])
    return after_doc(s, "import os  # notebook: paths from the environment")


def adaptP(s, name):
    """Q4_word_Prime scripts: the Q4_work modules are written next to them, so the sys.path line that pointed at Q4_work is dropped; the multiprocessing
    pool size comes from NPOOL (default = the original size). On Windows every pool worker re-imports the script, ~1.6 GB each for kinR, so 12 workers
    need ~20 GB of commit memory; p.map keeps the order, so the results do not depend on the pool size"""
    s = re.sub(r'sys\.path\.insert\(0, ROOT \+ r"\\Q4_work"\)(\s*;\s*|\n)', lambda m: "" if "\n" in m.group(1) else "", s)
    s = re.sub(r"\bPool\((\d+)\)", lambda m: f'Pool(int(os.environ.get("NPOOL", "{m.group(1)}")))', s)
    bad = [l for l in s.split("\n") if '\\Q4_work' in l and "submission_v10" not in l and 'os.environ.get("Q4_CACHE"' not in l and not l.lstrip().startswith("#")
           and '"""' not in l and "Q4_work/" not in l and "Q4_work\\cache is" not in l]
    doc = s[:s.index('"""', 3) + 3] if s.startswith('"""') else ""
    bad = [l for l in bad if l not in doc]
    assert not bad, (name, bad)
    return s


def local_imports(src):
    try: tree = ast.parse(src)
    except SyntaxError: return set()
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import): out |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0: out.add(n.module.split(".")[0])
    return out


def where(name):
    if os.path.exists(f"{P}\\{name}.py"): return "P"
    if os.path.exists(f"{Q}\\{name}.py"): return "Q"
    return None


md = lambda t: nbf.v4.new_markdown_cell(t.strip("\n"))
code = lambda t: nbf.v4.new_code_cell(t.strip("\n"))
WRITTEN, ORDER = set(), []


def wf(*names, sub=""):
    """%%writefile cells for these scripts and every local module they import (dependencies first), each written once"""
    cells = []
    def visit(n):
        key = sub + n
        if key in WRITTEN: return
        w = where(n); assert w, n; src = rd(f"{P if w == 'P' else Q}\\{n}.py"); WRITTEN.add(key)
        if not sub:
            for m in sorted(local_imports(src)):
                if m != n and where(m): visit(m)
        body = adaptP(src, n) if w == "P" else adaptQ(src, n); ORDER.append((sub + n, w))
        cells.append(code(f"%%writefile {sub}{n}.py\n" + body))
    for n in names: visit(n)
    return cells


cells = []
# =====================================================================================================================================================
cells.append(md(r"""
# CS-GY 6643 Project 3, Question 4: Penalty Kick Prediction (final notebook)

**Task.** From a pre-contact penalty clip, predict `kick_direction` and `goalkeeper_direction` (left / middle or "stands in the middle" / right, from the kicker's point of view). `goal_outcome` was removed from the score on 27 Oct, and the file predicts `goal` everywhere.
Kaggle metric = 0.7 · kick accuracy + 0.3 · keeper accuracy.

**Final file: `submission_N6.csv`** = N4's kick column + the G3 keeper. This notebook rebuilds it from the scripts that produced it and checks it byte for byte (Section 16).

**Results.** CV means source-grouped 5-fold on the 1,300 labelled public clips: every prediction comes from a model that never saw that clip's source. The public board scores 360 of the 900 test clips (±0.02).

| file | what changed | CV kick right (of 1,300, partition seeds 0-2) | CV keeper right | CV metric | Kaggle public |
|---|---|---|---|---|---|
| v10 (user's pipeline) | YOLO11x / RTMW / ViTPose++ pose trees and temporal networks, keeper CNN | 927 (0.713) | 1187 (0.913) | 0.773 | 0.77708 (old formula) |
| P4 | foot-prior ball, ground-plane camera, goal-plane keeper, MotionBERT channels in the networks | | | | 0.81472 |
| P4rkT | + ball on the NBJW spot where the ball was weak, keeper network, kicker re-pick | 953 (0.733) | 1200 (0.923) | 0.790 | 0.81861 |
| P4rkD | + pixel model: frozen ViTPose+ base tokens of three last-frame crops, camera-aware linear head, crops on the right kicker | 994 (0.765) | 1200 | 0.8122 | 0.83611 |
| N4 | + geometric blend of pose and pixel, corrected camera for 6 test clips | 996 | 1200 | 0.8132 | 0.83611 |
| **N6 (this notebook, the final submission)** | + keeper dip timing and "committed yet?" flags in the four keeper trees | **996** | **1203** | **0.8139** | 0.83361 |

Four partition seed sets (012 / 345 / 181920 / 212223) for N6: kick 996 / 1006 / 994 / 995, keeper 1203 / 1199 / 1198 / 1200, metric 0.8139 / 0.8184 / 0.8117 / 0.8127.

N6 and N4 differ only on 7 of the 900 test keepers.
* Out of fold, N6's changed keeper calls won 13 to 4 (pooled over the four sets).
* On the public board they lost 3 clips net, which is why N6's public score sits 0.0025 below N4's.
* Together, the expected private difference is under one keeper clip either way. N6 stays the final file by the rule written before any test feedback.

**How the notebook is organised.**
* "Method in brief" (next cell): the whole approach on one page.
* Sections 1-5: the perception pipeline from v7-v10.
* Sections 6-15: the new geometry and models, in the order they were built. Section 14 shows what the pixel model sees.
* Section 16: writes and checks the CSV.
* Section 17: the final assessment ("is the model guessing?", private vs public), with the confusion matrices and where the points are lost.
* Section 18: negative results and caveats.
* Section 19: the main challenges.

**Pretrained models (all frozen unless noted).**
* Ultralytics **YOLO11x** (COCO): persons and sports ball on 13 timed frames, test-time-augmented candidates, the ball on foot-centred crops (`ball2.py`), low-confidence persons for the kicker re-pick (`kid_exp.py`), ball near the spot (`contact_count.py`).
* **GroundingDINO-tiny** (`IDEA-Research/grounding-dino-tiny`): goal box.
* **RTMW-x** whole-body pose (`rtmw-dw-x-l_simcc-cocktail14_270e-384x288`, OpenMMLab, via `rtmlib` and ONNX Runtime): kicker and keeper, 133 keypoints including toes and heels.
* **RTMW3D-x** (`Soykaf/RTMW3D-x`, ONNX): its 2D output is a further kicker pose stream.
* **ViTPose++ base / large / huge** (`usyd-community/vitpose-plus-base`, `-large`, `-huge`, Hugging Face `transformers`):
  * kicker pose streams E / F / G;
  * keeper pose (large);
  * dense pose on every frame for 3D lifting (large);
  * the pixel model's token maps (base).
* **MotionBERT-Lite** (`walterzhu/MotionBERT`, `FT_MB_lite_MB_ft_h36m_global_lite`): 3D lifting of the dense kicker pose.
* **No Bells, Just Whistles** (NBJW; Gutiérrez-Pérez and Agudo, HRNetV2-W48 `SV_kp` / `SV_lines`): field calibration, used for the projected penalty spot and the kicker check.
* **WASB-SBDT** (Tarashima et al.): `mkball3.py` imports only the foot-reference helper of `wasb_ball.py`. The WASB detector itself was tested and is not used.
* **DINOv3 ViT-B/16** (`vit_base_patch16_dinov3.lvd1689m`): `zoom3.py` loads it. The final recipe uses only that script's meta file (clip order, usable / flip flags).
* **ResNet34** (`timm resnet34.a1_in1k`, ImageNet), **fine-tuned** as the keeper-direction CNN.

**External data:** none.

**Rules (PDF p. 12).**
* Every private clip is processed on its own: no pooling, normalisation or pseudo-labels across private clips. Statistics that need many clips, such as the camera prior and the ball-score threshold, come from public clips only.
* No manual labels and no human in the loop.
* No search for the original broadcasts or records.
* `public_position_clicks.csv` is used only to train and check the kicker / keeper picker.

**LLM usage.** Built with Claude Code, model Claude Opus 5.5 (Anthropic). The first pipeline (Sections 1-5) was built in an earlier Claude Code session that ran mainly on Claude Sonnet 5.5. Details: report, Section 5.
"""))

cells.append(md(r"""
## Method in brief
The model never sees the kick: every clip ends 0.02 s before the foot meets the ball, so everything below reads the approach and the last frames.

1. **Who is who.** YOLO11x finds the people and the ball. A ranker trained on the public position clicks picks the keeper, then the kicker (Section 1). A label-free check catches clips where the track ended on the referee: at contact the kicker must stand at the penalty spot, which the NBJW field calibration projects into the image. Those clips get the kicker re-picked (Sections 13 and 14).
2. **Body pose.** Five pose models (RTMW-x, RTMW3D-x, ViTPose++ base / large / huge) measure the kicker's joints on a set of frames covering the last second. ViTPose++ large also runs on every frame of that second for 3D lifting with MotionBERT-Lite (Sections 2, 3 and 8).
3. **Geometry.** A camera is fitted to the goal posts and the ball, so feet, hips and shoulders can be measured in metres on the ground and in the goal's frame. That makes behind-the-kicker, side-on, high and low views comparable (Sections 6 and 7).
4. **Kick direction: three readers.**
   * Gradient-boosting and extra-trees models on about 400 label-free kinematic features, one model per pose stream (Section 10).
   * Temporal 1D-convolution networks on the per-frame pose channels (Section 9).
   * A pixel model: frozen ViTPose+ base token maps of three crops of the last frame (body, torso-scaled body, legs), read by a camera-aware logistic head (Section 14). Most of its signal is the motion blur of the swinging leg.

   The pose readers are averaged, then combined with the pixel model by a geometric mean (Section 15).
5. **Keeper direction.** The keeper part has three members (Sections 4, 11 and 15):
   * trees on the keeper's movement in the goal plane, including when they last dipped and whether they have already committed;
   * a fine-tuned ResNet34 on goal-aligned keeper crops;
   * a temporal network.
6. **Validation.** Every number is source-grouped 5-fold CV on the 1,300 public clips, so a model never sees the source video of the clip it predicts. It is repeated on four partition seeds. Each change was adopted only by a rule written before its result (Sections 15 and 17).
"""))

cells.append(md(r"""
## 0. Setup and reproducibility
* `Q4data/train` = the 1,300 labelled public clips; `Q4data/val` = the 900 unlabelled test clips.
* **Clean room.** Each script is written by a `%%writefile` cell into `nbrun/` and run there as a subprocess. All paths come from the environment:
  * `Q4P_DIR` = `nbrun`, `Q4P_CACHE` = `nbrun/cache`, `Q4_CACHE` = `nbrun/wcache` (perception cache);
  * so the run never writes into `Q4_work` or into `Q4_word_Prime`'s own caches and CSVs.
* Running the stages as scripts also keeps the multiprocessing pools working on Windows.
* **Pool size.** The pool size of the feature scripts comes from the `NPOOL` environment variable. It defaults to 4; the original runs used 8–12, and the run saved here used 2 because the machine had less free memory. That run (6 Nov, 18:01–19:20) took 79 minutes with the heavy stages cached.
  * On Windows, every pool worker re-imports its script: about 1.6 GB each for `kinR.py`, so 12 workers need about 20 GB of commit memory.
  * `p.map` keeps the clip order, so the outputs do not depend on the pool size.
* **Two kinds of stage.**
  * *Heavy* stages read video or run large networks on the GPU: detection, pose, dense pose, MotionBERT, calibration, crops, ViTPose token maps, and the temporal networks (about 70 trainings). They take many GPU-hours, so they run only when their outputs are missing or `HEAVY=True`.
  * *Refit* stages are the feature tables, every tree and keeper fit, the pixel classifier, and every CSV step. With `REFIT=True` they are recomputed even when cached. The older output moves to `nbrun/_prev/<time>/` and the two are compared, so every refit doubles as a reproduction check.
* **Seeding the run folder.** `nb_prep.ps1` copies the caches of the original run into `nbrun/` (about 27 GB) and reads its sources only:
  * `wcache` ← `Q4_work/cache`;
  * `cache` ← `Q4_word_Prime/cache`, without the experiment crops and embeddings the final recipe never reads;
  * `cache_kf`, `cache_g`, and the MotionBERT / NBJW / WASB code and weights.

  With empty folders, every stage runs from the raw videos in the order shown.
* GPU training is not bit-for-bit deterministic, so retraining the networks (`HEAVY=True`) can change a few test predictions. The CPU fits are seeded and deterministic.
* The cross-validation runs behind the results table are not repeated here. Their out-of-fold files are read from the cache by Section 17.
* Time is handled by timestamp, not frame index: fps ranges from 15 to 60.
"""))
cells.append(code(r"""
import os, sys, time, pickle, shutil, subprocess, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
PROJECT_ROOT = r"C:\path\to\project-root"
NB_DIR = os.path.join(PROJECT_ROOT, "Q4_word_Prime")                     # this notebook and the submitted CSVs
RUN = os.path.join(NB_DIR, "nbrun")                                     # clean-room working folder of this run
CACHE, WCACHE, KC, GC = (os.path.join(RUN, d) for d in ("cache", "wcache", "cache_kf", "cache_g"))
FINAL = "N6"                    # "N4" stops before the keeper change of 6 Nov
REFIT = True                    # recompute every refit stage even when cached (older output -> nbrun/_prev, compared)
HEAVY = False                   # also recompute the GPU / video stages (many GPU-hours)
for d in (RUN, CACHE, WCACHE, os.path.join(RUN, "logs"), os.path.join(RUN, "nb_src")): os.makedirs(d, exist_ok=True)
os.chdir(RUN)
os.environ.update(Q4_ROOT=PROJECT_ROOT, Q4_CACHE=WCACHE, Q4P_DIR=RUN, Q4P_CACHE=CACHE, HF_HUB_OFFLINE="1", PYTHONWARNINGS="ignore", PYTHONIOENCODING="utf-8",
                  NPOOL=os.environ.get("NPOOL", "4"))    # worker processes per feature script (originally 8-12); each re-imports its script, ~1.6 GB for kinR. Results do not depend on it
print("feature-script pool size NPOOL =", os.environ["NPOOL"])
E_KF = dict(Q4P_CACHE=KC, Q4_OVL=os.path.join(CACHE, "kfixW"))           # the re-picked kicker chain (Section 13)
E_B3 = dict(BALLF="ball3", CAMF="camera2_ball3")                         # the NBJW-spot ball chain (Section 12)
PREV = os.path.join(RUN, "_prev", time.strftime("%Y%m%d_%H%M%S")); LOG = []
CFG = [(12, 96), (15, 128), (20, 96), (20, 128), (30, 128)]               # the five configurations of every network bag
LOGF = open(os.path.join(RUN, "logs", "notebook_run.log"), "a", encoding="utf-8")


def say(*a):   # print, and keep a copy in nbrun/logs/notebook_run.log (readable while the notebook runs)
    print(*a); print(time.strftime("[%H:%M:%S]"), *a, file=LOGF, flush=True)


def run(script, *args, env=None, isolated=False, tail=6):
    t0 = time.time(); cmd = [sys.executable] + (["-I", os.path.join(RUN, script)] if isolated else [script]) + [str(a) for a in args]
    r = subprocess.run(cmd, cwd=RUN, env={**os.environ, **(env or {})}, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = [l for l in (r.stdout + r.stderr).splitlines() if l.strip() and "Warning" not in l and "deprecated" not in l]
    say(f"$ python {'-I ' if isolated else ''}{script} {' '.join(map(str, args))} {env or ''}  [{time.time() - t0:.0f}s, rc {r.returncode}]")
    say("\n".join("    " + l[:300] for l in out[-tail:]))
    if r.returncode: raise RuntimeError(f"{script} failed:\n" + "\n".join(out[-30:]))


path = lambda p: p if os.path.isabs(p) else os.path.join(RUN, p)
present = lambda p: (len(os.listdir(path(p))) > 0) if os.path.isdir(path(p)) else os.path.exists(path(p))


def _diff(x, y):
    if isinstance(x, (np.ndarray, np.generic)) or isinstance(y, (np.ndarray, np.generic)):
        x, y = np.asarray(x), np.asarray(y)
        if x.shape != y.shape: return np.inf
        if x.dtype.kind in "fc" and y.dtype.kind in "fc":
            a, b = x.astype(float), y.astype(float); m = ~(np.isnan(a) & np.isnan(b)); d = np.abs(a - b)[m]
            return 0.0 if d.size == 0 else (np.inf if np.isnan(d).any() else float(d.max()))
        return 0.0 if np.array_equal(x, y) else np.inf
    if isinstance(x, dict) and isinstance(y, dict): return np.inf if set(x) != set(y) else max([_diff(x[k], y[k]) for k in x] or [0.0])
    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)): return np.inf if len(x) != len(y) else max([_diff(a, b) for a, b in zip(x, y)] or [0.0])
    if isinstance(x, pd.DataFrame): return 0.0 if x.equals(y) else np.inf
    if isinstance(x, float) and isinstance(y, float): return 0.0 if (x == y or (np.isnan(x) and np.isnan(y))) else abs(x - y)
    try: return 0.0 if x == y else np.inf
    except Exception: return np.inf


def compare(old, new):
    ext = os.path.splitext(new)[1]
    if ext == ".csv": return "identical bytes" if open(old, "rb").read() == open(new, "rb").read() else f"DIFFERENT: {int((pd.read_csv(old) != pd.read_csv(new)).any(axis=1).sum())} rows"
    if ext == ".npy": a, b = np.load(old, allow_pickle=True), np.load(new, allow_pickle=True); a, b = (a.item(), b.item()) if a.dtype == object and a.shape == () else (a, b)
    elif ext == ".npz": za, zb = np.load(old, allow_pickle=True), np.load(new, allow_pickle=True); a, b = {k: za[k] for k in za.files}, {k: zb[k] for k in zb.files}
    elif ext == ".pkl": a, b = pickle.load(open(old, "rb")), pickle.load(open(new, "rb"))
    else: return "identical bytes" if open(old, "rb").read() == open(new, "rb").read() else "different bytes"
    d = _diff(a, b); return "identical" if d == 0 else f"max abs difference {d:.2e}"


def stage(name, outputs, fn, heavy=False):
    # heavy stages run only when an output is missing (or HEAVY); refit stages also when REFIT, with the previous output kept and compared
    missing = [o for o in outputs if not present(o)]
    if not missing and not (HEAVY if heavy else REFIT): say(f"[{name}] cached ({len(outputs)} outputs)"); return
    moved = {}
    if not heavy:
        for o in outputs:
            if present(o) and os.path.isfile(path(o)):
                dst = os.path.join(PREV, os.path.relpath(path(o), RUN)); os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.move(path(o), dst); moved[o] = dst
    say(f"[{name}] {'recomputing' if moved else 'computing'}" + (f" (missing: {missing[:4]})" if missing else "")); t0 = time.time(); fn()
    for o, dst in moved.items(): r = compare(dst, path(o)); LOG.append((name, o, r)); say(f"    check {o}: {r} vs the previous output")
    say(f"[{name}] done in {time.time() - t0:.0f}s")


print("run folder:", RUN, "| perception dirs cached:", [d for d in ("det", "ballspot", "kpts", "kpts3d", "kptsE", "kptsF", "kptsG", "kkeep", "kkeepF") if present(os.path.join(WCACHE, d))])
"""))

# ---------------- 1-5: perception (Q4_work) ----------------
cells.append(md(r"""
## 1. Persons, goal and kicker / keeper tracks (v7, user's pipeline)
* `extract.py`: YOLO11x on 13 frames at fixed offsets before the last frame (1.0 … 0 s), with a torso-colour descriptor per box.
* `track.py` and `rank2.py`: the v6 pair ranker and its tracks, the fallback track set.
* `scene.py`: the GroundingDINO-tiny goal box, plus YOLO11x on an upscaled crop of the goal mouth (small keepers are missed full-frame).
* `scene3.py`: YOLO11x with test-time augmentation at confidence 0.05.
* `rank5.py`: picks the keeper first, then the kicker given the keeper. It uses ball-to-feet and goal-box features, and is trained on the clicks out of fold by source. Held-out agreement with the clicks: keeper 99.8%, kicker 99.5%.
* `track5.py`: follows each pick over the 13 frames and fills gaps with TTA detections. `apply_tracks.py` writes `tracks.pkl`.
"""))
S1 = ("extract", "track", "rank2", "scene", "scene3", "rank5", "track5", "apply_tracks"); cells += wf(*S1)
cells.append(code(r"""
stage("tracks", ["wcache/tracks.pkl"], lambda: [run(s + ".py") for s in %S1%], heavy=True)
""".replace("%S1%", repr(S1))))
cells.append(md(r"""
## 2. Frame set, camera stabilisation, RTMW-x keypoints and the old ball (v5-v7)
* `ballspot.py` builds the frame set F per clip: the last 8 native frames plus sparse frames back to 1 s.
  * It chains frame-to-frame homographies (LK tracks on the background, with the kicker masked) to the last frame. Every position below is in last-frame coordinates.
* `kpts.py`: RTMW-x whole-body pose on the kicker box of every frame.
  * Its left/right labels are unreliable at ~100 px person height, so every kicker feature is label-free.
  * The foot that travels farther is the kicking foot, and body lines are axes taken mod 180°.
* `ballmed.py`, `blob_ball.py` and `apply_ball.py`: the v7 ball, used as one candidate source of the new ball.
"""))
S2 = ("ballspot", "kpts", "ballmed", "blob_ball", "apply_ball"); cells += wf(*S2)
cells.append(code(r"""
stage("frames + RTMW", ["wcache/ballspot", "wcache/kpts"], lambda: (run("ballspot.py"), run("kpts.py")), heavy=True)
stage("old ball", ["wcache/ball_final.pkl"], lambda: [run(s) for s in ("ballmed.py", "blob_ball.py", "apply_ball.py")], heavy=True)
"""))
cells.append(md(r"""
## 3. Pose streams: five for the kicker, two for the keeper (v6-v10)
Each pose model is an independent measurement of the same joints.
* The trees use one stream each (A = RTMW-x, F = ViTPose++ large, G = ViTPose++ huge).
* The temporal networks train on several streams of the same clip as separate samples and average them at test time (measurement-noise augmentation). Their stream groups are A+B+E+F and A+B+G, where B = RTMW3D-x and E = ViTPose++ base.
* Keeper streams: RTMW-x (`kkeep`) and ViTPose++ large (`kkeepF`).
"""))
cells += wf("kpts3d", "kpts_vit", "kpts_keeper", "kpts_vit_keeper")
cells.append(code(r"""
stage("RTMW3D", ["wcache/kpts3d"], lambda: run("kpts3d.py"), heavy=True)
for out, model in (("kptsE", "base"), ("kptsF", "large"), ("kptsG", "huge")):
    stage(f"ViTPose++ {model}", [f"wcache/{out}"], lambda out=out, model=model: run("kpts_vit.py", env=dict(VIT_OUT=out, VIT_MODEL=f"usyd-community/vitpose-plus-{model}")), heavy=True)
stage("keeper RTMW", ["wcache/kkeep"], lambda: run("kpts_keeper.py"), heavy=True)
stage("keeper ViTPose-L", ["wcache/kkeepF"], lambda: run("kpts_vit_keeper.py", env=dict(VIT_OUT="kkeepF", VIT_MODEL="usyd-community/vitpose-plus-large")), heavy=True)
"""))
cells.append(md(r"""
## 4. Keeper CNN (v5)
* `align.py` cuts goal-aligned keeper crops: each is rotated so the goal direction points up, which makes a horizontal flip an exact mirror.
* `final_cnn.py` fine-tunes ResNet34 on all 1,300 public clips:
  * crops at 0 s and 0.25 s, flip with label swap, 30 epochs, 5 seeds;
  * flip TTA on the test clips.
* Its out-of-fold probabilities (`oof_gk_resnet34.a1_in1k_0.npy`, from v5's `train_cnn.py`) are read from the cache by the keeper blend's CV.
"""))
cells += wf("align", "final_cnn")
cells.append(code(r"""
stage("keeper CNN", ["wcache/cnn_gk_private.npy"], lambda: (run("align.py"), run("final_cnn.py")), heavy=True)
"""))
cells.append(md(r"""
## 5. Feature modules from v4-v10 (imported below)
* `kin2.py`: 259 label-free kinematic kicker features in the stabilised, ball-anchored goal frame.
* `seqfeat.py`: the per-frame channels of the temporal network.
* `kinkeep.py`: 162 camera-stabilised keeper features.

The mirror is exact in these frames: negate the lateral axis, flip the foot sign, swap left and right. It doubles the training data and is used as test-time augmentation.
"""))
cells += wf("kin2", "seqfeat", "kinkeep")
# the pose scripts that kfix.py / kidfix.py rerun for a few clips are read from nb_src/
cells.append(md("The pose scripts that the kicker re-pick (Section 13) and the identity fix (Section 14) rerun for a few clips are read from `nb_src/`, as in the original run:"))
cells += wf("ballspot", "kpts", "kpts_vit", "kpts3d", sub="nb_src/")

# ---------------- 6. ball and camera ----------------
cells.append(md(r"""
## 6. The stationary ball with a foot prior, and a ground-plane camera from the goal posts
* **Ball** (`ball2.py`).
  * The clip ends 0.02 s before contact, so the kicking foot must be at the ball. Yet the v10 ball was more than 0.5 kicker heights from every foot in 225 clips.
  * Candidates: YOLO11x sports-ball detections on foot-centred crops of every frame, plus the old balls.
  * Each candidate cluster is scored as support × confidence × exp(−(d_foot / 0.3)²).
  * Result: 244 balls moved, and the far-from-feet clips fell from 225 to 24.
* **Posts** (`goalposts.py`): LSD segments inside the goal box, merged into posts.
* **Camera** (`camera2.py`): orthographic, with zero roll, elevation θ, azimuth φ and scale s. It is fitted to ball → goal-line centre = 11 m and post → post = 7.32 m.
  * Ground coordinates of an image point X at height z: g = A⁻¹(X − b_g − a_z z).
  * Clips without posts use the public-median θ.
* `common.py`, `cvlib.py` and `streams.py` are the shared helpers: paths, the CV protocol, and the pose streams in one keypoint layout.
"""))
cells += wf("common", "cvlib", "streams", "runner", "ball2", "goalposts", "camera2")
cells.append(code(r"""
stage("ball2", ["cache/ball2.pkl"], lambda: run("ball2.py"), heavy=True)
stage("goal posts", ["cache/goalposts.pkl"], lambda: run("goalposts.py"), heavy=True)
stage("camera2", ["cache/camera2.pkl"], lambda: run("camera2.py"))
"""))
cells.append(md(r"""
## 7. Kicker features: kin2 with the new ball, and ground-plane features (kinR)
* `kin2b.py` recomputes the v10 kin2 features with the new ball.
* `kinR.py` adds 141 features and 36 per-frame channels in metres on the ground. It rectifies only what one elevated view can measure:
  * ground-contact points (feet);
  * same-height differences (hip, shoulder and ear axes);
  * displacements of roughly constant-height points (hip, shoulder and head trajectories).
* Segment leans stay as image leans, re-weighted by the azimuth.
"""))
cells += wf("kin2b", "kinR")
cells.append(code(r"""
for S in "AFG": stage(f"kin2 {S}", [f"cache/kin2_{S}_ball2.pkl"], lambda S=S: run("kin2b.py", S, "ball2"))
for S in "AFG": stage(f"kinR {S}", [f"cache/kinR_{S}.pkl"], lambda S=S: run("kinR.py", S))   # the trees' table; seq2b.py computes the same channels per stream itself
"""))
cells.append(md(r"""
## 8. MotionBERT-Lite 3D kicker pose (network channels)
* `kpts_dense.py`: ViTPose++ large on every native frame of the last ~1 s.
* `lift_mb.py`: MotionBERT-Lite lifts the dense pose to 3D. It runs with `python -I`, and the model code is loaded by explicit path.
* `probe_mb.py`: checks the output frame (handedness, z scale).
* `kinMB.py`: rotates the lifted pose into the goal frame with the clip's camera, giving 29 per-frame channels for the networks.
"""))
cells += wf("kpts_dense", "lift_mb", "probe_mb", "kinMB")
cells.append(code(r"""
stage("dense ViTPose-L", ["cache/kptsD"], lambda: run("kpts_dense.py"), heavy=True)
stage("MotionBERT", ["cache/mb3d"], lambda: run("lift_mb.py", isolated=True), heavy=True)
stage("MotionBERT frame check", [], lambda: run("probe_mb.py"))
stage("kinMB", ["cache/kinMB.pkl"], lambda: run("kinMB.py"))
"""))
cells.append(md(r"""
## 9. Temporal pose networks (two bags × two stream groups)
* `seq2b.py` builds the per-frame channels:
  * `b2`: the v6 channels with the new ball;
  * `b2RM`: plus the 36 ground-plane channels, 3 camera statics, and the MotionBERT goal-frame channels.
* `train_seqP.py` is the v6 / v10 network:
  * an input projection, three residual dilated 1D convolutions, and a head on [last frame, mean of the last 6, max] plus statics;
  * AdamW with one-cycle LR, label smoothing 0.1, mirror augmentation;
  * a clip's pose streams train as separate samples and are averaged at test time.
* The kick network = mean of 20 members: {b2, b2RM} × {stream groups EF, G} × 5 configurations, each over 3 seeds, fitted on all public clips (`MODE=full`).
"""))
cells += wf("seq2b", "train_seqP")
cells.append(code(r"""
for S in "ABEFG":
    stage(f"seq {S}", [f"cache/seq_{S}_b2.pkl", f"cache/seq_{S}_b2RM.pkl"], lambda S=S: (run("seq2b.py", S), run("seq2b.py", S, "RM")))
NN_MAIN = [f"cache/prvP_{e}_{h}_st{st}_{q}.npy" for q in ("b2", "b2RM") for st in ("EF", "G") for e, h in CFG]
stage("kick networks", NN_MAIN, lambda: [run("train_seqP.py", e, h, env=dict(SEQ=q, STREAMS=st, MODE="full"), tail=1) for q in ("b2", "b2RM") for st in ("EF", "G") for e, h in CFG], heavy=True)
"""))
cells.append(md(r"""
## 10. Kick trees
`trees_private.py` fits one tree model per pose stream (RTMW, ViTPose-L, ViTPose-H) on [kin2 with the new ball, kinR]. Each model is:
* a gradient-boosting bag (4 configurations × 3 seeds) averaged with ExtraTrees (500 trees, min leaf 3, × 3 seeds);
* fitted on all 1,300 public clips with mirror rows, and read on the test clips with mirror TTA.
"""))
cells += wf("trees_private")
cells.append(code(r"""
for S in "AFG": stage(f"kick trees {S}", [f"cache/prv_trees_{S}_b2R.npy"], lambda S=S: run("trees_private.py", f"{S}_b2R", f"kin2_{S}_ball2", f"kinR_{S}"))
"""))
cells.append(md(r"""
## 11. Keeper
* `keep2b.py`: the v10 keeper features with the new ball, for the RTMW and ViTPose-L keeper keypoints. It also prints their CV.
* `keepR.py`: goal-plane keeper features. Image motion is decomposed on the camera's (along the goal line, up) directions, a system that is well conditioned in every view.
* `keep_private.py` (`KEEP_TAG=b2G`) builds the 7-member keeper and fits it on all public clips:
  * 2 × (RTMW trees) + 2 × (ViTPose-L trees) + RTMW goal-plane trees + ViTPose-L goal-plane trees + the keeper CNN.
* Keeper network:
  * `keepseq.py` builds the keeper per-frame channels;
  * `train_seqP.py` with `TARGET=keeper` trains 5 configurations × seed sets {0,1,2} and {3,4,5} on all public clips;
  * the mean of the 10 runs is `pg_private_NNk.npy`.
* N4's keeper = (7 · tree/CNN keeper + keeper network) / 8, written by `make_subP4k.py` (Section 12).
* N6 adds `keepD.py` to the four tree members (Section 15).
"""))
cells += wf("keep2b", "keepR", "keep_private", "keepseq")
cells.append(code(r"""
stage("keeper features (new ball)", ["cache/keep_A_b2.pkl", "cache/keep_F_b2.pkl"], lambda: run("keep2b.py"))
stage("goal-plane keeper features", ["cache/keepR_A.pkl", "cache/keepR_F.pkl"], lambda: run("keepR.py"))
stage("keeper trees + CNN, private", ["cache/pg_private_b2G.npy"], lambda: run("keep_private.py", env=dict(KEEP_TAG="b2G")))
stage("keeper sequences", ["cache/seq_A_kp.pkl", "cache/seq_F_kp.pkl"], lambda: run("keepseq.py"))
NN_KEEP = [(e, h, sd) for sd in ("0,1,2", "3,4,5") for e, h in CFG]
KEEP_FILES = [f"cache/prvP_{e}_{h}_st_kp{'' if sd == '0,1,2' else '_sd345'}_keeperAF.npy" for e, h, sd in NN_KEEP]
stage("keeper network", KEEP_FILES, lambda: [run("train_seqP.py", e, h, env=dict(TARGET="keeper", BASE="AF", SEQ="kp", STREAMS="", MODE="full", SEEDS=sd), tail=1) for e, h, sd in NN_KEEP], heavy=True)
def nnk():   # pipeline23.py's mean of the 10 keeper-network runs, columns (left, right, stands)
    Qk = sum(np.load(path(f)) for f in KEEP_FILES) / len(KEEP_FILES); np.save(path("cache/pg_private_NNk.npy"), Qk[:, [0, 2, 1]]); say("    keeper network counts (left / right / stands)", np.bincount(Qk[:, [0, 2, 1]].argmax(1), minlength=3))
stage("keeper network mean", ["cache/pg_private_NNk.npy"], nnk)
"""))
cells.append(md(r"""
## 12. First CSVs (P4, P4k) and the ball on the NBJW penalty spot (P4b3k)
* `make_subP.py` writes P4: kick = 0.5 · mean(three tree streams) + 0.5 · mean(20 network members); keeper = the 7-member keeper.
* `make_subP4k.py` adds the keeper network, giving the keeper = (7 · tree/CNN keeper + keeper network) / 8.
* **Ball 3.**
  * `nbjw_cam.py` (No Bells Just Whistles field calibration, `python -I`) gives an independent ball estimate: the penalty spot projected into the last frame.
  * Where ball2 is weak (score in the public bottom third) and more than 1 m from the spot, `mkball3.py` uses the spot. This needs a usable calibration, and the spot must be nearer the feet than ball2. 79 clips moved.
* The whole kick chain is rebuilt with that ball: camera, kin2, kinR, kinMB, sequences, trees and networks.
* `make_subP4b3.py` uses those predictions only on the moved test clips.
"""))
cells += wf("make_subP", "make_subP4k", "nbjw_cam", "kinN", "wasb_ball", "mkball3", "make_subP4b3")
cells.append(code(r"""
stage("P4", ["submission_P4.csv", "cache/Pk_private_P4.npy"], lambda: run("make_subP.py", "P4", "b2+b2RM", "A_b2R", "F_b2R", "G_b2R", env=dict(KEEPER="b2G", COMPARE=os.path.join(NB_DIR, "submission_P4.csv"))))
stage("P4k", ["submission_P4k.csv", "cache/pg_private_b2GNNk.npy"], lambda: run("make_subP4k.py"))
stage("NBJW calibration", ["cache/nbjw"], lambda: run("nbjw_cam.py", isolated=True), heavy=True)
stage("ball3", ["cache/ball3.pkl"], lambda: run("mkball3.py"))
stage("camera (ball3)", ["cache/camera2_ball3.pkl"], lambda: run("camera2.py", "ball3"))
for S in "AFG":
    stage(f"kin2 / kinR {S} (ball3)", [f"cache/kin2_{S}_ball3.pkl", f"cache/kinR_{S}_camera2_ball3.pkl"], lambda S=S: (run("kin2b.py", S, "ball3"), run("kinR.py", S, env=dict(CAMF="camera2_ball3"))))
stage("kinMB (ball3)", ["cache/kinMB_b3.pkl"], lambda: run("kinMB.py", env=dict(CAMF="camera2_ball3")))
for S in "ABEFG": stage(f"seq {S} (ball3)", [f"cache/seq_{S}_b2_b3.pkl", f"cache/seq_{S}_b2RM_b3.pkl"], lambda S=S: (run("seq2b.py", S, env=E_B3), run("seq2b.py", S, "RM", env=E_B3)))
for S in "AFG": stage(f"kick trees {S} (ball3)", [f"cache/prv_trees_{S}_b3R.npy"], lambda S=S: run("trees_private.py", f"{S}_b3R", f"kin2_{S}_ball3", f"kinR_{S}_camera2_ball3"))
NN_B3 = [f"cache/prvP_{e}_{h}_st{st}_{q}.npy" for q in ("b2_b3", "b2RM_b3") for st in ("EF", "G") for e, h in CFG]
stage("kick networks (ball3)", NN_B3, lambda: [run("train_seqP.py", e, h, env=dict(SEQ=q, STREAMS=st, MODE="full"), tail=1) for q in ("b2_b3", "b2RM_b3") for st in ("EF", "G") for e, h in CFG], heavy=True)
stage("P4b3k", ["submission_P4b3k.csv", "cache/Pk_private_P4b3.npy"], lambda: run("make_subP4b3.py", env=dict(ARM="b3")))
"""))
cells.append(md(r"""
## 13. Kicker re-pick (P4rkT)
* **Check.** `kickcheck.py` measures, on the ground plane, the distance from the tracked kicker's last-frame box to the NBJW-projected spot. It is label-free: at contact − 0.02 s the kicker is at the spot.
* **Re-pick.** Where the tracked box is more than 6 m away and another detected person is within 2 m, `kfix.py pick` takes that person and follows them back over the 13 frames. 32 clips: 23 public, 9 test.
  * `kfix.py pose` / `pose3d` rerun the pose models on those clips into the overlay `cache/kfixW`.
* The kick chain is rebuilt for them in `cache_kf/`:
  * dense pose and MotionBERT;
  * ball2 for the re-picked clips, merged into a copy;
  * camera2, with every other clip put back to the main camera, because its no-posts prior moves when any public clip changes;
  * kin2, kinR, kinMB and the sequences.
* `make_subP4rk.py` with `ARM=T` gives the re-picked test clips the `cache_kf` trees. That choice was made after the CV: its networks did not gain.
"""))
cells += wf("blend_eval", "kickcheck", "kfix", "make_subP4rk")
cells.append(code(r"""
stage("kicker check", ["cache/kickcheck.pkl"], lambda: run("kickcheck.py"))
stage("kicker re-pick", ["cache/kfixW/tracks.pkl", "cache/kfix_clips.pkl"], lambda: run("kfix.py", "pick"))
stage("re-picked pose", ["cache/kfixW/kpts", "cache/kfixW/kptsG", "cache/kfixW/kpts3d"], lambda: (run("kfix.py", "pose"), run("kfix.py", "pose3d")), heavy=True)


def kf_copy_and_heavy():   # pipeline34.py: start cache_kf from the main cache, then dense pose, MotionBERT and ball2 for the re-picked clips only
    KF = set(pickle.load(open(path("cache/kfix_clips.pkl"), "rb"))); os.makedirs(KC, exist_ok=True)
    for f in ("goalposts.pkl", "ball2.pkl"):
        if not os.path.exists(f"{KC}/{f}"): shutil.copy(f"{CACHE}/{f}", f"{KC}/{f}")
    for d in ("kptsD", "mb3d"):
        os.makedirs(f"{KC}/{d}", exist_ok=True)
        for fn in os.listdir(f"{CACHE}/{d}"):
            if fn[:-4] not in KF and not os.path.exists(f"{KC}/{d}/{fn}"): shutil.copy(f"{CACHE}/{d}/{fn}", f"{KC}/{d}/{fn}")
    run("kpts_dense.py", env=E_KF); run("lift_mb.py", isolated=True, env=E_KF)
    src = open(path("ball2.py"), encoding="utf-8").read()
    src = "import sys; sys.path.insert(0, r'%s')\n" % RUN + src.replace("for n, c in enumerate(CLIPS):", f"for n, c in enumerate([c for c in CLIPS if c in {sorted(KF)!r}]):", 1)
    src = src.replace('pickle.dump(out, open(f"{PC}/ball2.pkl", "wb"))', 'full = pickle.load(open(f"{PC}/ball2.pkl", "rb")); full.update(out); pickle.dump(full, open(f"{PC}/ball2.pkl", "wb"))', 1)
    assert "full.update(out)" in src; open(f"{KC}/_ball2_kf.py", "w", encoding="utf-8").write(src); run(f"{KC}/_ball2_kf.py", env=E_KF)
stage("re-pick chain: dense pose, MotionBERT, ball2", ["cache_kf/kptsD", "cache_kf/mb3d", "cache_kf/ball2.pkl"], kf_copy_and_heavy, heavy=True)


def kf_camera():           # camera2 in cache_kf, then every clip that was not re-picked gets the main camera back (kidfix.py's code for the same step)
    run("camera2.py", env=E_KF); KF = set(pickle.load(open(path("cache/kfix_clips.pkl"), "rb")))
    new, old = pickle.load(open(f"{KC}/camera2.pkl", "rb")), pickle.load(open(f"{CACHE}/camera2.pkl", "rb"))
    for c in old:
        if c not in KF: new[c] = old[c]
    pickle.dump(new, open(f"{KC}/camera2.pkl", "wb")); say(f"    camera2 put back for {len(old) - len(KF)} clips that were not re-picked")
stage("re-pick chain: camera", ["cache_kf/camera2.pkl"], kf_camera)
for S in "AFG": stage(f"re-pick chain: kin2 / kinR {S}", [f"cache_kf/kin2_{S}_ball2.pkl", f"cache_kf/kinR_{S}.pkl"], lambda S=S: (run("kin2b.py", S, "ball2", env=E_KF), run("kinR.py", S, env=E_KF)))
stage("re-pick chain: kinMB", ["cache_kf/kinMB.pkl"], lambda: run("kinMB.py", env=E_KF))
for S in "ABEFG": stage(f"re-pick chain: seq {S}", [f"cache_kf/seq_{S}_b2.pkl", f"cache_kf/seq_{S}_b2RM.pkl"], lambda S=S: (run("seq2b.py", S, env=E_KF), run("seq2b.py", S, "RM", env=E_KF)))
for S in "AFG": stage(f"re-pick chain: kick trees {S}", [f"cache_kf/prv_trees_{S}_b2R.npy"], lambda S=S: run("trees_private.py", f"{S}_b2R", f"kin2_{S}_ball2", f"kinR_{S}", env=E_KF))
NN_KF = [f"cache_kf/prvP_{e}_{h}_st{st}_{q}.npy" for q in ("b2", "b2RM") for st in ("EF", "G") for e, h in CFG]   # read by make_subP4rk.py; ARM=T uses only the trees
stage("re-pick chain: networks", NN_KF, lambda: [run("train_seqP.py", e, h, env=dict(E_KF, SEQ=q, STREAMS=st, MODE="full"), tail=1) for q in ("b2", "b2RM") for st in ("EF", "G") for e, h in CFG], heavy=True)
stage("P4rkT", ["submission_P4rkT.csv", "cache/Pk_private_P4rkT.npy"], lambda: run("make_subP4rk.py", env=dict(ARM="T")))
"""))
cells.append(md(r"""
## 14. Pixel model (P4rkD)
A frozen ViTPose+ base reads three last-frame crops. The crops are cut on the kicker and mirrored so that the goal is on one side:
* body on the hip mid;
* torso;
* legs, 1.2 kicker heights.

The token maps (8 × 6 × 768 per crop) are joined, standardised and reduced by PCA 512. A camera-aware logistic head then reads them: tokens × (1, cos φ, sin φ) × (1, foot sign), C = 0.001, with mirror rows and mirror TTA.
* `contact_count.py run` writes `contact.pkl`, which `late_reader.py` reads on import.
* `late_reader.py` gives the calibration-based goal side used to mirror the crops.
* `zoom3.py` writes the clip order and the usable / flip flags (`zoom3_meta.npz`).
* `crops8.py` and `crops5.py` cut the crops. `vitpose_emb.py` turns them into token maps.
* **Crops on the right kicker.** `kid.py` and `kid_exp.py` (sweep → combo → tta → v2) build a motion-gated kicker track from the anchor pick, with a re-pick of the fastest approacher near the spot.
  * `kidfix.py prep` / `pose` put the 82 clips whose kicker track changes (53 public, 29 test) into the overlay `cache/kidW`.
  * `fix_crops.py D` and `fix_emb.py D` recut and re-embed the 46 whose last-frame kicker box moved: 31 public, 15 test.
* `make_subPx2.py` fits the pixel model on all public clips. It writes P4rkD = 0.5 · P4rkT + 0.5 · pixel, the input of N4.
"""))
cells += wf("contact_count", "late_reader", "zoom3", "crops8", "crops5", "vitpose_emb", "kid", "kid_exp", "kidfix", "fix_crops", "vp_embed", "fix_emb", "pxlib", "make_subPx2")
cells.append(code(r"""
stage("ball at the spot (contact)", ["cache/contact.pkl"], lambda: run("contact_count.py", "run"), heavy=True)
stage("crop meta (zoom3)", ["cache/zoom3_meta.npz"], lambda: run("zoom3.py"), heavy=True)
stage("crops", ["cache/crops8.npy", "cache/crops8_meta.npz", "cache/crops5.npy", "cache/crops5_meta.npz"], lambda: (run("crops8.py"), run("crops5.py")), heavy=True)
for v in ("body_hm", "body_tor", "legs12"): stage(f"ViTPose+ base tokens {v}", [f"cache/ce_{v}_vpbase.npy"], lambda v=v: run("vitpose_emb.py", v, "base"), heavy=True)
stage("kicker identity tracks", ["cache/kid_combo.pkl"], lambda: [run("kid_exp.py", *a) for a in (("sweep", "x"), ("combo",), ("tta",), ("v2",))], heavy=True)
stage("identity overlay", ["cache/kidW/tracks.pkl", "cache/kid_clips.pkl"], lambda: run("kidfix.py", "prep"))
stage("identity overlay pose", ["cache/kidW/kpts", "cache/kidW/kptsE"], lambda: run("kidfix.py", "pose"), heavy=True)
stage("crops on the right kicker", ["cache/cropsfix_D.npz"], lambda: run("fix_crops.py", "D", env=dict(Q4_OVL=os.path.join(CACHE, "kidW"))))
stage("tokens on the right kicker", [f"cache/ce_{v}_vpbaseD.npy" for v in ("body_hm", "body_tor", "legs12")], lambda: run("fix_emb.py", "D"))
PX = "e_body_hm_vpbaseD,e_body_tor_vpbaseD,e_legs12_vpbaseD:512:0.001"
stage("pixel model + P4rkD", ["cache/Pp_e_body_hm_vpbaseD+e_body_tor_vpbaseD+e_legs12_vpbaseD_512_0.001.npy", "submission_P4rkD.csv", "cache/Pk_private_P4rkD.npy"], lambda: run("make_subPx2.py", "P4rkD", "0.5", PX))
"""))
cells.append(md(r"""
**What the pixel model sees.** Below are the three crops for four labelled public clips:
* the pixel model's most confident correct call of each kick class;
* its most confident miss of a middle kick.

The calls are out of fold. The leg's motion blur in the last frame carries most of the signal. In the last frame a middle kick can look just like a side kick (bottom row), which is why middles are the main loss. No test clip is shown.
"""))
cells += wf("fig_crops")
cells.append(code(r"""
from IPython.display import Image, display
run("fig_crops.py", tail=6); display(Image(filename=os.path.join(RUN, "logs", "fig", "crops.png")))
"""))
cells.append(md(r"""
## 15. N4 (geometric blend, corrected camera) and N6 (keeper dip timing)
Rules for both were written before their CV results (`RULES_nov4.md`, `RULES_nov6.md`).
* **Camera below the horizon.** `nov4_cam.py make` targets the clips whose fitted camera has sin φ < 0 (12 public, 6 test):
  * when camera2's goal side disagrees with the calibration flip, it uses φ + 180°, otherwise −φ;
  * then it rebuilds kinR on that camera.
  * `nov4_private.py trees` reads the main trees on the corrected rows, and `nov4_private.py cam` the pixel model's camera terms. It also reads the 25 fps grid rows (`pipeline_grid.py features`, `timegrid.py`) for an arm the final recipe does not use.
* **Geometric blend.** `make_subN4.py N4 --geo --cam`: kick = normalise(√(base · pixel)), where base = P4rkT's.
  * Source-grouped CV: +2 / +3 / +3 clips over the linear blend; log loss is better on every set.
* **Keeper dip timing.** `keepD.py` reads the keeper's goal-plane track:
  * the time since the last up/down turning point of the hips and head, and the depth of the last dip;
  * "committed yet?" flags: lateral move > 0.10 m in the last 0.12 s, feet raised > 0.10 m.
* `nov6_private.py keeper` appends these to the four keeper tree members and refits them on all public clips. It checks that without keepD it reproduces the N4 keeper exactly.
  * CV: +3 / +2 / +2 / +2 keeper clips on four partition sets, adopted by the rule.
* `make_subN6.py N6 --g3` = N4's kick + the new keeper. It changes 7 of 900 test keepers.
"""))
cells += wf("nov4_cam", "timegrid", "pipeline_grid", "nov4_private", "make_subN4", "keepD", "nov6_private", "make_subN6")
cells.append(code(r"""
stage("corrected camera", ["cache/camera2f.pkl"] + [f"cache/kinR_{S}_camera2f.pkl" for S in "AFG"], lambda: run("nov4_cam.py", "make"))
def grid_rows():   # pipeline_grid.py "features": the 25 fps grid rows in cache_g (read by nov4_private.py trees for an arm N4 does not use)
    os.makedirs(GC, exist_ok=True); EG = dict(Q4P_CACHE=GC, Q4_GRID="1")
    for f in ("ball2.pkl", "camera2.pkl"): shutil.copy(f"{CACHE}/{f}", f"{GC}/{f}")
    for S in "AFG": run("kin2b.py", S, "ball2", env=EG); run("kinR.py", S, env=EG)
stage("grid rows", [f"cache_g/kin2_{S}_ball2.pkl" for S in "AFG"] + [f"cache_g/kinR_{S}.pkl" for S in "AFG"], grid_rows)
stage("trees on corrected rows", [f"cache/nov4/prv_trees_{S}.npz" for S in "AFG"], lambda: run("nov4_private.py", "trees"))
stage("pixel on corrected camera", ["cache/nov4/prv_px_cam.npy"], lambda: run("nov4_private.py", "cam"))
stage("N4", ["submission_N4.csv", "cache/Pk_private_N4.npy"], lambda: run("make_subN4.py", "N4", "--geo", "--cam"))
stage("keeper dip timing", ["cache/keepD_A.pkl", "cache/keepD_F.pkl"], lambda: run("keepD.py"))
stage("keeper with dip timing, private", ["cache/nov6/pg_private_b2GD.npy"], lambda: run("nov6_private.py", "keeper"))
if FINAL == "N6": stage("N6", ["submission_N6.csv", "cache/pg_private_N6.npy"], lambda: run("make_subN6.py", "N6", "--g3"))
"""))
# ---------------- 16. CSV ----------------
cells.append(md(r"""
## 16. The CSV and its checks
This section checks three things:
* the format, against `submission_template.csv`: same clips, same order, valid labels;
* that the file is byte-identical to the submitted file;
* every reproduction check of the refit stages above.
"""))
cells.append(code(r"""
sub = pd.read_csv(os.path.join(RUN, f"submission_{FINAL}.csv")); tpl = pd.read_csv(os.path.join(PROJECT_ROOT, "Q4data", "submission_template.csv"))
assert list(sub.columns) == list(tpl.columns) and list(sub["clip"]) == list(tpl["clip"]), "clips / columns differ from the template"
assert sub["kick_direction"].isin(["left", "middle", "right"]).all() and sub["goalkeeper_direction"].isin(["left", "right", "stands in the middle"]).all()
print(sub.shape, "| kick", sub.kick_direction.value_counts().to_dict(), "| keeper", sub.goalkeeper_direction.value_counts().to_dict(), "| outcome", sub.goal_outcome.value_counts().to_dict())
for name in ("P4", "P4k", "P4b3k", "P4rkT", "P4rkD", "N4", "N6"):     # every CSV of the chain against the file of the original run (P4, P4rkT, P4rkD, N4 and N6 were submitted)
    a, b = os.path.join(RUN, f"submission_{name}.csv"), os.path.join(NB_DIR, f"submission_{name}.csv")
    if os.path.exists(a) and os.path.exists(b): print(f"submission_{name}.csv rebuilt here vs the original file: {compare(b, a)}")
tag = lambda r: "same" if r.startswith("identical") else "~   " if r.startswith("max abs") and float(r.split()[-1]) <= 1e-4 else "DIFF"
n = {k: sum(tag(r) == k for _, _, r in LOG) for k in ("same", "~   ", "DIFF")}
print(f"\n{len(LOG)} refit checks (older output vs recomputed): {n['same']} identical | {n['~   ']} within 1e-4 (floating-point rounding: summation order in the tree ensembles,"
      f" multithreaded BLAS in the pixel model's PCA and logistic fit) | {n['DIFF']} larger")
for name, o, r in LOG: print(f"   {tag(r)} {name:42s} {o:70s} {r}")
print(sub.head())
"""))
# ---------------- 17. assessment ----------------
cells.append(md(r"""
## 17. Final assessment: is the model guessing, and how close will private be to public?
`final_assess.py` and `assess_shuffle.py` use only the out-of-fold predictions of the 1,300 labelled clips, across four partition sets. Results of the 6 Nov run (`logs/final/assess.txt`):

* **Not guessing.**
  * Labels reshuffled 100,000 times with the predictions fixed: metric 0.443 ± 0.010, against our 0.814 (38 SD). No shuffle came close.
  * Reshuffled within each source: 34 SD.
  * Retrained on shuffled labels, the ViTPose-L kick trees fall from 0.689 to 0.42–0.45.
  * Accuracy rises steadily with confidence: kick 0.55 → 0.92 across confidence fifths.
* **Against `chance.csv`.**
  * `chance.csv` (public 0.52277) is the PDF's "educated guess": kick and keeper to the kicker's natural side. It scores 0.5205 on the labelled clips.
  * On the same clips we win 396 kick clips to its 52 (McNemar p ≈ 1e-67).
  * Every subgroup is ahead: fps, resolution, foot, gender, camera side, clicked source, class.
  * Of 100,000 simulated 540-clip private boards, ours leads on 100%, with a smallest margin of +0.19.
* **Private vs public.**
  * The 360 / 540 split alone gives SD(public − private) = 0.022 for a model with no overfitting. A gap within 1% of public is therefore only 30% likely (within 1 point, 35%), whatever the model.
  * Forecast given N6's public 0.83361: private ≈ 0.820 ± 0.017 (80% range 0.798–0.842), 95% chance of at least 0.792.
  * The forecast is below public because public sits 1.7 SD above the CV level (probably luck), and 900 × level = 360 × public + 540 × private.
"""))
cells += wf("p4rkt", "ovf_lib", "final_assess", "assess_shuffle")
cells.append(code(r"""
RUN_ASSESS = True
if RUN_ASSESS:
    os.makedirs(os.path.join(RUN, "logs", "final"), exist_ok=True); run("final_assess.py", tail=60); run("assess_shuffle.py", tail=6)
"""))
cells.append(md(r"""
**Figures: N6 out of fold** (partition seeds 0-2; labelled public clips only).
* **Confusion matrices.**
  * Left and right kicks are read well: 90% and 86% recall.
  * Middle kicks are mostly called as sides: 8% recall.
  * Keepers who stand are mostly called as divers.
* **Confidence.** Kick accuracy rises from 0.55 in the least confident fifth to 0.92 in the most confident.
* **Lost points.** About half of all lost metric points are missed middle kicks.
"""))
cells += wf("fig_results")
cells.append(code(r"""
run("fig_results.py", tail=8)
for f in ("confusion.png", "confidence.png"): display(Image(filename=os.path.join(RUN, "logs", "fig", f)))
"""))
# ---------------- 18. negatives / caveats ----------------
cells.append(md(r"""
## 18. What did not work, and caveats
Each test below was decided by a rule written before its result, judged on source-grouped CV with fresh partition seeds. One kick clip = 0.0005 of the metric.
* **Kick cues that did not help:**
  * a follow-through or swing direction (AUC ≤ 0.55);
  * the ball's direction after contact, read from the last frames;
  * plant- and kicking-foot pointing cues;
  * run-up rhythm (`kinX.py`, +4 over four sets: below the bar);
  * head tilt and kicking-foot shape (`kinH.py`, −5);
  * middle-kick stackers on ViTPose internals (−3 to −7);
  * P(middle) scaling;
  * a 15 fps time grid, CLAHE, and degraded-crop training for 640×360;
  * Sapiens pose, DINOv2 / DINOv3 / MViT pixel features (all below ViTPose+ base);
  * view augmentation, keypoint smoothing, a common time grid;
  * LightGBM, stackers, other blend weights.
* **Keeper:** a pixel model, an early-mover split, and stands reweighting.
* **Outcome:** no model beat always predicting `goal`.
* **Where the model is weak:**
  * about half of the lost points are missed middle kicks: middle recall is about 0.08, and every component ranks middles at AUC 0.61–0.66;
  * a further quarter come from left/right errors where the pose and pixel models agree and are both wrong;
  * 640×360 clips (0.63 on 21 labelled clips) and unclicked sources (0.76 against 0.84) are weaker.
* **Caveats:**
  * Ball and keypoint quality are self-consistency checks, not labelled accuracy, and nothing was hand-corrected.
  * The tests looked at many variants, but each was decided on fresh seeds by a rule fixed in advance, and the four-set spread of the final recipe is 0.003.
  * Public (360 clips) is about 2 points above the CV level, so expect private around 0.82.
"""))
cells.append(md(r"""
## 19. Challenges
* **The kick is never shown.**
  * Clips stop 0.02 s before contact, so the direction has to be read from the approach and the last frames.
  * Middle kicks (14% of the labelled clips) and keepers who stand (5%) are mostly decided at or after contact. They remain the main loss: 48% of the lost points are missed middles (Section 17).
* **Small, blurred players.**
  * In broadcast video a kicker is often about 100 px tall, with motion blur and compression.
  * Pose models swap left and right joints at that size. So every kicker feature is label-free (the foot that travels farther is the kicking foot), and five pose models serve as independent measurements.
* **The wrong person.**
  * In some clips the track ended on the referee or another player near the spot.
  * The label-free spot check found 32 such clips. The kicker re-pick and identity fix repaired them (Sections 13 and 14), and the pixel crops had to be recut on the right kicker.
* **Very different camera views.**
  * Behind-the-kicker, side-on, high and low angles change what a left kick looks like.
  * The ground-plane camera and the camera-aware pixel head deal with this. Six test clips whose fitted camera sat below the horizon needed a corrected azimuth (Section 15).
* **Frame rates from 15 to 60 fps.**
  * Features use time, not frame counts.
  * One test source (31 clips at 15 fps and 640×360) is unlike anything in training and is read less well.
* **Unseen sources.**
  * Almost every test source is new, so a random clip split would overstate accuracy.
  * Every decision used source-grouped CV, a rule fixed before its result, and fresh partition seeds.
* **Little data.**
  * There are 1,300 labelled clips.
  * The pixel model gains about 2.8 points of kick accuracy per doubling of its training clips, so more labelled clips would be the strongest lever. No suitable external dataset was available.
* **Compute.**
  * Detection and pose take many GPU-hours.
  * On Windows every multiprocessing worker re-imports its script, so the notebook caches the heavy stages and limits the pool size (Section 0).
"""))

nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, f"{P}\\Q4_Penalty_Kick_final.ipynb")
print("cells", len(cells), "| scripts written", len(ORDER), "| from Q4_work", sum(w == "Q" for _, w in ORDER), "| from Q4_word_Prime", sum(w == "P" for _, w in ORDER))
print("order:", ", ".join(n for n, _ in ORDER))
