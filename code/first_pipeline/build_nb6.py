"""Assemble Q4_Penalty_Kick.ipynb (v5 pipeline) from the stage scripts + report text."""
import nbformat as nbf, re
R = r"C:\path\to\project-root\Q4_work"
src = lambda n: open(f"{R}\\{n}", encoding="utf-8").read()


def adapt(s):
    s = s.replace('ROOT = r"C:\\path\\to\\project-root"', "ROOT = PROJECT_ROOT")
    s = s.replace('ROOT + r"\\Q1-Q3work\\yolo11x.pt"', "YOLO_DET").replace('ROOT + r"\\Q4_word_GPT\\models\\yolo11m-pose.pt"', "YOLO_POSE")
    s = "\n".join(l for l in s.split("\n") if "if len(sys.argv) > 1" not in l)          # CLI test limits: not for the notebook
    s = s.replace("from kin2 import slope\n", "")                                      # slope() is already defined by the kin2 cell
    s = s.replace("import kin2\nfrom kin2 import legs_at, match2, slope\n", "")
    s = s.replace("import kin2\nfrom kin2 import legs_at, match2\n", "").replace("CT = kin2.CT", "CT = 2.0").replace("import seqfeat\n", "").replace("seqfeat.build(", "build(")
    return s


def wrap(s, cond):
    return f"if {cond}:\n" + "\n".join(("    " + l if l.strip() else l) for l in s.split("\n"))


md = lambda t: nbf.v4.new_markdown_cell(t.strip("\n"))
code = lambda t: nbf.v4.new_code_cell(t.strip("\n"))
cells = []

cells.append(md(r"""
# CS-GY 6643 Project 3, Question 4: Penalty Kick Prediction

**Goal.** From a pre-contact penalty clip (it ends 0.02 s before foot-ball contact) predict `kick_direction`, `goalkeeper_direction` and `goal_outcome`. Score = 0.60·acc(kick) + 0.25·acc(keeper) + 0.15·acc(outcome). Left/right are from the kicker's point of view.

**Results**

| version | what changed | kick | keeper | outcome | CV score | Kaggle public LB |
|---|---|---|---|---|---|---|
| v2 | pose-feature GBDT on single-frame snapshots | 0.53 | 0.79 | 0.72 | 0.62 | 0.61555 |
| v3 | better kicker/keeper picker, keeper CNN | 0.58 | 0.87 | 0.72 | 0.67 | 0.65555 |
| v4 | **label-free kinematic kick features** (RTMW wholebody, stationary ball, gravity-tilt) | 0.64 | 0.87 | 0.72 | 0.71 | 0.71555 |
| v5 | kick ensemble (GBDT+ExtraTrees); **camera-stabilised keeper features** + CNN | 0.652 | 0.896 | 0.722 | 0.724 | 0.72250 |
| **v6 (this notebook)** | kick = 0.5·trees + 0.5·**temporal pose network** (two pose streams as measurement-noise augmentation) | **0.680** | **0.896** | 0.722 | **0.740** | **0.75916** |

CV = source-grouped 5-fold (validation on sources never seen in training). For v5, CV and the public leaderboard agreed within 0.002; v6 scored 0.019 *above* its CV estimate, which is within the noise of a 360-clip public split (about ±0.02), so I would not read it as more than a good sign. The best public scores in the class are about 0.78, so a small gap remains (Section 9).

**Pretrained models / external resources (named as the rules require).**
* Ultralytics **YOLO11x** (COCO person + sports-ball detector) and **YOLO11m-pose** (older stages), frozen.
* **RTMW-x wholebody pose** (`rtmw-dw-x-l_simcc-cocktail14_270e-384x288`, OpenMMLab, run through `rtmlib` + ONNX Runtime), frozen: 133 keypoints including toes/heels. Downloaded from `download.openmmlab.com`.
* **RTMW3D-x** (`Soykaf/RTMW3D-x`, Hugging Face, ONNX), frozen: only its *2D* keypoints are used, as an independent second measurement of the same joints.
* **ResNet34 (`resnet34.a1_in1k`, timm)**, ImageNet-pretrained, fine-tuned for keeper direction.
* Tried but **not in the final submission**: RTMW3D's depth output, ViTPose++ base (`usyd-community/vitpose-plus-base`), DINOv3 ViT-B/16, MViTv2-S (Kinetics-400).
* No external datasets. No manual labelling, no use of private labels, no cross-clip use of private predictions, and I did not search for the original broadcasts or any record of these penalties.

**LLM usage.** Done with Claude Code (Anthropic) in an agentic session. Chat history: *(attach the shareable link / PDF here before submitting, as the project's LLM policy requires)*.
"""))

cells.append(md(r"""
## 0. Setup and reproducibility notes
* `Q4data/train` = the 1,300 labelled *public* clips, `Q4data/val` = the 900 unlabelled *private* clips (folder names differ from the README).
* Every stage writes to `Q4_work/cache` and **skips work that is already cached**, so the notebook re-runs quickly; delete `cache/` (or set `FORCE_CNN=True`) to recompute from raw video. From scratch it takes a few hours on one laptop GPU (RTX 5070 Ti): detection ≈ 30 min, ball/homography ≈ 30 min, wholebody keypoints ≈ 20 min, keeper CNN ≈ 12 min. GPU training is not bit-for-bit deterministic, so a from-scratch run may change a few predictions: re-running the temporal-network bag changed 5 of 900 kick predictions relative to the file I submitted (submission_v6.csv; the rerun is saved as submission_v6_notebook_rerun.csv).
* `RUN_OPTIONAL_EVAL=True` also re-runs the keeper-CNN cross-validation and the final out-of-fold evaluation (≈ 20 min).
* ONNX Runtime needs the CUDA/cuDNN libraries that PyTorch bundles, so `import torch` must come *before* `rtmlib` (done in the pose cells); otherwise it silently falls back to CPU (10x slower).
* Paths are Windows paths; edit `PROJECT_ROOT`, `YOLO_DET`, `YOLO_POSE` for another machine.
* Time handling: frames are chosen by **timestamp**, not index (fps varies: 15, 24, 25, 30, 50, 60).
"""))
cells.append(code(r"""
import os, sys, warnings
warnings.filterwarnings("ignore")
PROJECT_ROOT = r"C:\path\to\project-root"
YOLO_DET  = PROJECT_ROOT + r"\Q1-Q3work\yolo11x.pt"
YOLO_POSE = PROJECT_ROOT + r"\Q4_word_GPT\models\yolo11m-pose.pt"
FORCE_CNN = False          # retrain the keeper CNN and the temporal-network bag even if their predictions are cached
RUN_OPTIONAL_EVAL = False  # re-run keeper-CNN cross-validation + final out-of-fold evaluation
os.makedirs(PROJECT_ROOT + r"\Q4_work\cache", exist_ok=True)
"""))

cells.append(md(r"""
## 1. Data observations that shaped the method
* Broadcast footage, mostly 1280x720, players only ~60-120 px tall. Clips are short (median 1.9 s, 99% under 5.5 s); the last ~second is nearly all there is.
* The goal direction varies by camera, so **image-left is not the kicker's left**. Everything is expressed in a **goal-facing frame**: u = ball→keeper direction, v = kicker's left. A left/right mirror is then exact (negate v, flip preferred-foot sign, swap labels): 2x augmentation and test-time augmentation.
* The ball is stationary for essentially the whole clip, so it gives a fixed anchor for the geometry once camera motion is removed.
* 215 sources; 842 of 900 private clips come from sources never seen in training, so **all validation is grouped by `source_id`**.
* `public_position_clicks.csv` (879 clips) is used only to train and score the kicker/keeper picker, never as an input for private clips.
"""))

cells.append(md("## 2. Stage 1 - person detection at timed offsets (YOLO11x)\nFor every clip, 13 frames at fixed offsets before the last frame (1.0 … 0 s) are run through YOLO11x with a torso-colour descriptor per box."))
cells.append(code(adapt(src("extract.py"))))

cells.append(md(r"""
## 3. Stage 2 - find the kicker and keeper
`track.py`: a ranker scores each person box as kicker/keeper at the anchor frame (0.25 s before the end) - 86.7% / 83.4% correct on held-out sources. `rank2.py` scores **kicker-keeper pairs** jointly (adds each candidate's motion over time, distance to the ball, pair geometry, colour difference): **kicker 96.4%, keeper 95.8%** on held-out sources (the ranker is trained on clicks, out-of-fold by source; clips without clicks and all private clips use the model fit on all clicked clips).
"""))
cells.append(code(adapt(src("track.py"))))
cells.append(code(adapt(src("rank2.py"))))

cells.append(md(r"""
## 4. Stage 3 - frame set, camera compensation and the stationary ball
`ballspot.py` builds a frame set per clip (last 8 native frames + sparse frames back to 1 s), estimates a **homography between consecutive frames** (LK tracks on the background, kicker masked) and chains them to the last frame. Ball candidates (YOLO11x ball class on kicker-centred crops, upscaled) are warped into last-frame coordinates and the **stationary ball is the position most frames agree on** (support-weighted, with a prior near the kicker's feet).

Measured on the training clips (no ball ground truth exists, so these are self-consistency checks): ball supported by ≥3 frames in **93.5%** of clips (≥5 frames in 86%); in the last frame the nearest foot keypoint is within 0.5 body-heights of the ball in **84.8%** of clips (within 1.0 in 91.8%); low-support clips are much less consistent (58% within 0.5 h), which is where this stage is weakest.
"""))
cells.append(code(adapt(src("ballspot.py"))))

cells.append(md(r"""
## 5. Stage 4 - wholebody keypoints (RTMW-x) and a measurement problem I had to design around
RTMW-x (133 keypoints incl. big toe, small toe, heel) is run on the kicker's box in every frame (box re-run once from the keypoints if it was off); the same is done for the keeper in `kpts_keeper.py`.

Two properties of this model at ~100 px person height shaped the features:
1. **Its confidence scores are on a 2-5 scale** (not 0-1), so thresholds must be set on that scale.
2. **Left/right keypoint labels are unreliable**: in 32% of clips the two ankles swap identity somewhere in a 6-frame window, and in 18% the toe/heel group belongs to the other ankle. So "right ankle = kicking foot" is unsafe. I therefore build features **label-free**: the two feet are re-grouped physically (ankle-anchored), tracked over the last frames, and the foot that **travels farther is the kicking foot** (the other is the plant foot); hip and shoulder lines are used as axes (angle mod 180°). `preferred_foot` enters only as a sign feature. (The foot that moves more matches `preferred_foot` in about 66-73% of clips, which is partly pose noise and partly the label; I did not rely on it.)
"""))
cells.append(code(adapt(src("kpts.py"))))
cells.append(code(adapt(src("kpts_keeper.py"))))
cells.append(md("`kpts3d.py` runs **RTMW3D-x** on the same kicker boxes. Its 2D keypoints are a second, partly independent measurement of the same joints (used in Section 7b as noise augmentation); its depth output was tested and did not help (Section 9)."))
cells.append(code(adapt(src("kpts3d.py"))))

cells.append(md(r"""
## 6. Stage 5 - kinematic features (the main gain)
`kin2.py` computes 259 features per clip in the stabilised, ball-anchored goal frame (unit = kicker height): plant- and kicking-foot placement relative to the ball, foot angle, kicking-foot velocity, shank/thigh angles and **angular velocities**, hip/shoulder/ear axes (mod 180°) and their rates, trunk and head lean, approach path (hip position/velocity relative to the ball over 0.8 s), and **tilt of every leg segment, the trunk and the head relative to gravity** expressed on the goal axes, with rates over the last frames.

Ablation (source-grouped, 3 seeds; ≈ ±0.01 noise): alone, the plant leg's shank/thigh block gives 0.556 and dropping it costs −0.025; dropping trunk −0.016, hip axis −0.012, approach −0.012, head −0.007. **Foot heel-to-toe angle adds nothing** (dropping it even helps by 0.005): at this resolution foot orientation is mostly noise, and the signal is in leg and trunk *lean*. The gravity-tilt block alone (0.622) already matches the whole earlier feature set (0.617); all together 0.631-0.641, versus 0.565 for the previous snapshot pose features.
"""))
cells.append(code(adapt(src("kin2.py"))))

cells.append(md(r"""
## 7. Stage 6 - keeper features (camera-stabilised) and the keeper CNN
The earlier keeper model measured keeper motion in raw image coordinates, but broadcast cameras pan, which corrupts a subtle dive-onset signal. `kinkeep.py` computes the same goal-frame features from **homography-stabilised keeper keypoints**: hip/feet/shoulder/head displacement over time and its velocity/acceleration, stance width, body lean relative to gravity, arm position, crouch/stretch from the box. Keeper accuracy: old features 0.832 → **0.885 (GBDT) / 0.889 (ExtraTrees)**; averaged with the ResNet34 keeper CNN (goal-aligned keeper crops, 0 s and 0.25 s, flip+label-swap augmentation, 30 epochs) it reaches **0.896**.
"""))
cells.append(code(adapt(src("kinkeep.py"))))
cells.append(code(adapt(src("align.py"))))
cells.append(md(r"""
## 7b. Stage 6b - temporal pose network (the second big kick gain)
The tree models see hand-picked summaries of the pose. `seqfeat.py` instead exports the **per-frame label-free channels** for the last 12 frames (positions of kicking/plant foot, knee, hip, hip-mid, shoulder-mid and nose in the ball-anchored goal frame, hip/shoulder/ear axes mod 180°, gravity tilt of six body segments, time offset, validity) and `train_seq.py` trains a small **1D temporal-convolution network** on them with exact mirror augmentation (original or mirrored sample, label swapped) and mirror test-time averaging.

Two things made it work well:
1. **Measurement-noise augmentation.** Each clip has two independent pose measurements (RTMW, and RTMW3D's 2D keypoints; `seqfeatB.py`). The network trains on both as separate samples with the same label and averages both at test time: 0.652 → 0.658 alone, and the ensemble with the trees 0.670 → 0.679.
2. **Short training.** 12-20 epochs beat 60-100 (0.667-0.671 vs 0.633-0.650): the network overfits quickly with 1,300 clips.
Source-grouped CV for kick: bag of 5 network configurations 0.672 alone; **trees + network (0.5/0.5) 0.680** (left/right-only accuracy 0.776), against 0.652 for the trees. Variants that did *not* help: feeding the 259 hand-built features into the network (0.640 alone, worse ensemble), GRU (0.632) and transformer (0.621) encoders, seed bagging inside a fold.
"""))
cells.append(code(adapt(src("seqfeat.py"))))
cells.append(code(adapt(src("seqfeatB.py"))))
tc = adapt(src("train_cnn.py")).replace("arch, target, EP, folds = sys.argv[1], sys.argv[2], int(sys.argv[3]), [int(x) for x in sys.argv[4].split(\",\")]",
                                          "arch, target, EP, folds = 'resnet34.a1_in1k', 'gk', 30, [0, 1, 2, 3, 4]").replace("seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0", "seed = 0")
cells.append(md("The next two cells (optional, `RUN_OPTIONAL_EVAL`) cross-validate the keeper CNN and evaluate the final v5 configuration out of fold."))
cells.append(code(wrap(tc, "RUN_OPTIONAL_EVAL")))
cells.append(code(wrap(adapt(src("final_eval5.py")), "RUN_OPTIONAL_EVAL")))
ts_lines = [l for l in src("train_seq.py").replace('ROOT = r"C:\\path\\to\\project-root"', "ROOT = PROJECT_ROOT").split("\n") if "oof_ens_lab2" not in l and not l.startswith("base = (M[")]
ts_src = "\n".join(ts_lines)
assert (chr(39) * 3) not in ts_src and "EP = int(sys.argv[1])" in ts_src
cv_seq = "import sys, os\nTRAIN_SEQ_SRC = r\'\'\'" + ts_src + "\'\'\'\nif RUN_OPTIONAL_EVAL:\n    for ep, h in [(12, 96), (15, 128), (20, 96), (20, 128), (30, 128)]:\n        sys.argv = ['train_seq.py', str(ep), str(h)]; os.environ.update(KIN='0', SPF='1', ARCH='conv', AUGB='1')\n        exec(compile(TRAIN_SEQ_SRC, 'train_seq', 'exec'), {'__name__': 'seq_cv', 'PROJECT_ROOT': PROJECT_ROOT})"
cells.append(md("Optional: cross-validate the temporal network (5 configurations) and evaluate the final v6 kick/keeper configuration out of fold."))
cells.append(code(cv_seq))
fe6 = adapt(src("final_eval6.py")).replace('M = pickle.load(open(f"{C}/oof_ens_lab2.pkl", "rb")); trees = (M["hgb259"] + M["extratrees"]) / 2', "trees = (Ph + Pe) / 2")
fe6 = fe6.replace('KL = pickle.load(open(f"{C}/oof_keep_lab.pkl", "rb")); Pg = (KL["new_hgb"] + KL["new_et"] + KL["cnn"]) / 3; ', "")
cells.append(code(wrap(fe6, "RUN_OPTIONAL_EVAL")))

cells.append(md(r"""
## 8. Stage 7 - final training on all public clips, inference on private clips, CSV
* Keeper CNN: 5 seeds × 30 epochs on all 1,300 clips, flip-TTA predictions on the 900 private clips.
* Kick = mean of {GBDT bag (4 configs × 3 seeds), ExtraTrees (3 seeds)} on the 259 kinematic features; keeper = mean of {GBDT, ExtraTrees on the 162 stabilised keeper features, CNN}; outcome = always `goal`. All tree models train on original + mirrored samples and average original/mirrored predictions at test time.
`make_sub5` writes `submission_v5.csv` (the previous best); the last cell writes `submission_v6.csv` and checks it against `submission_template.csv`.
"""))
cells.append(code(wrap(adapt(src("final_cnn.py")), 'FORCE_CNN or not os.path.exists(PROJECT_ROOT + r"\\Q4_work\\cache\\cnn_gk_private.npy")')))
cells.append(code(adapt(src("make_sub5.py"))))
cells.append(md(r"""
**v6:** the temporal pose network bag (5 configurations x 3 seeds, trained on all 1,300 clips on both pose streams, original + mirrored, predictions averaged over streams and mirror) is blended 50/50 with the v5 tree probabilities for kick; the keeper is unchanged. This writes `submission_v6.csv`, the final submission.
"""))
cells.append(code(wrap(adapt(src("final_seq.py")), 'FORCE_CNN or not os.path.exists(PROJECT_ROOT + r"\\Q4_work\\cache\\nn_private.npy")')))
cells.append(code(adapt(src("make_sub6.py"))))

cells.append(md(r"""
## 9. What did not work, challenges, and honest caveats
**Negative results** (source-grouped CV kick accuracy unless noted):
* *Image models for kick:* frozen DINOv3 features 0.46-0.50, fine-tuned ResNet34 on kicker crops 0.49-0.50, MViTv2-S video model 0.51 (constant prediction), and an **end-to-end ResNet34 + temporal-conv model on homography-stabilised, ball-centred, goal-aligned 12-frame clips** (0.475 on fold 0 where the trees score 0.671; training loss only fell from 1.03 to 0.89). Three differently built image models all failed to learn beyond chance, while the same information expressed as pose geometry works, so with ~1,300 clips the cue (body lean) is not learnable from pixels with these recipes.
* *3D lifting* (RTMW3D-x depth): 99 depth-aware lean features give 0.575 alone but only +0.004 on top of the 2D kinematics. Camera-pitch scale and depth sign made no difference.
* *Second pose measurement stream* (RTMW3D's own 2D keypoints): no gain, fused or averaged (0.649-0.652).
* *Ensembling* gave about +1 point (GBDT 0.641, ExtraTrees 0.648, mean 0.652); MLP 0.622 and logistic regression 0.562 were worse, so the cue is nonlinear.
* *Keeper→kick stacking* (the keeper's predicted direction as a kick feature, or a multiplicative prior): no gain (0.642 vs 0.641).
* *`middle` kicks* (13.9% of kicks): a mirror-symmetric signed score thresholded at ±a gives middle recall 0.25 only at a large loss of left/right accuracy; I still essentially never predict `middle` (recall ≈ 0.07). With the network bag the ensemble does predict some `middle` (40 clips out-of-fold, 40% correct against a 14% base rate; recall 0.09), but left/right-only accuracy (0.776) remains the number that matters for kick.
* *Goal outcome*: even when predicted kick and keeper sides agree and the kick is confident (>0.9), only 56% are no_goal (n=36), so the best prediction stays `goal` (0.722).

**Where the model is weak.** Confidence is well calibrated: accuracy is 0.86 on the quarter of clips with the highest kick confidence and 0.45 on the lowest quarter. The 107 clips predicted with confidence >0.8 are 87% correct, and the confident errors are not concentrated in any camera type or frame rate. The ball locator is weakest on the 6.5% of clips with <3 supporting frames.

**Caveats.** Detection accuracy is measured only where ground truth exists (kicker/keeper clicks on 879 training clips); ball and keypoint quality are self-consistency checks, not labelled accuracy, and nothing was hand-corrected on private clips. Model choices (ExtraTrees leaf size, blend of three keeper models, CNN epochs) were made on the same out-of-fold predictions, so the CV score is mildly optimistic; the independent check is the leaderboard, which agrees (v5: 0.7238 vs 0.7225; v6: CV 0.740 vs public 0.759, a gap inside the ±0.02 noise of the 360-clip public split). Public scores near 0.78 exist (v6 is at 0.759), so I may be missing some of what the best teams use; I do not know what it is.
"""))

nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, f"{R}\\Q4_Penalty_Kick.ipynb"); print("cells", len(cells))
