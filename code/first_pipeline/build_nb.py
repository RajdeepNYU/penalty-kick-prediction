"""Assemble Q4_Penalty_Kick.ipynb from the pipeline stage scripts + report text."""
import nbformat as nbf
R = r"C:\path\to\project-root\Q4_work"
src = lambda n: open(f"{R}\\{n}", encoding="utf-8").read()


def adapt(s):
    s = s.replace('ROOT = r"C:\\path\\to\\project-root"', "ROOT = PROJECT_ROOT")
    s = s.replace('ROOT + r"\\Q1-Q3work\\yolo11x.pt"', "YOLO_DET").replace('ROOT + r"\\Q4_word_GPT\\models\\yolo11m-pose.pt"', "YOLO_POSE")
    s = s.replace("if len(sys.argv) > 1: todo = todo[:int(sys.argv[1])]\n", "")
    return s


HEADER = '''import pickle, pandas as pd
ROOT = PROJECT_ROOT; D = ROOT + r"\\Q4data"; C = ROOT + r"\\Q4_work\\cache"
_f = pickle.load(open(f"{C}/feats.pkl", "rb"))
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); gt = gt[gt["clip"].isin(_f)].reset_index(drop=True)
X = np.stack([_f[c][0] for c in gt["clip"]]); Xm = np.stack([_f[c][1] for c in gt["clip"]])
SW = {"left": "right", "right": "left", "middle": "middle", "stands in the middle": "stands in the middle"}
'''


def selfcontained(s):
    return adapt(s).replace('sys.argv = ["x"]\nfrom cv import X, Xm, gt, SW, C\n', HEADER)


def wrap(s, cond):
    return f"if {cond}:\n" + "\n".join(("    " + l if l.strip() else l) for l in s.split("\n"))


md = lambda t: nbf.v4.new_markdown_cell(t.strip("\n"))
code = lambda t: nbf.v4.new_code_cell(t.strip("\n"))
cells = []

cells.append(md(r"""
# CS-GY 6643 Project 3, Question 4: Penalty Kick Prediction

**Goal.** From a pre-contact penalty clip (it ends 0.02 s before foot-ball contact) predict `kick_direction`, `goalkeeper_direction` and `goal_outcome`. Score = 0.60·acc(kick) + 0.25·acc(keeper) + 0.15·acc(outcome). Left/right are from the kicker's point of view.

**Results (source-grouped 5-fold CV, i.e. validation on sources never seen in training).**

| version | kick | keeper | outcome | est. score | Kaggle public LB |
|---|---|---|---|---|---|
| v2 (earlier pipeline) | 0.53 | 0.79 | 0.72 | 0.62 | **0.61555** (CV predicted 0.62) |
| **v3 (this notebook)** | **0.58** | **0.87** | 0.72 | **≈ 0.67** | *(not yet submitted when this was written)* |

* The gain from v2 to v3 comes mainly from **a much better kicker/keeper picker** (kicker found 86.7% → 96.4%, keeper 83.4% → 95.8%, on held-out sources) and from a **keeper CNN trained longer** on the corrected crops (0.66 → 0.865 alone).
* Kick direction is still the weak spot (≈0.58 vs 0.47 majority and 0.50 preferred-foot baseline). Goal outcome is structurally capped near 0.72 (see Section 6), so it is always `goal`.
* All numbers are cross-validation estimates; the v2 public score matched its CV estimate, but the v3 score is still unverified on Kaggle.

**Pretrained models / external resources used (named as the rules require).**
* Ultralytics **YOLO11x** (COCO person + sports-ball detector) and **YOLO11m-pose** (COCO keypoints), frozen.
* **ResNet34 (`resnet34.a1_in1k`, timm)**, ImageNet-pretrained, fine-tuned for keeper direction.
* Tried but **not in the final submission**: DINOv3 ViT-B/16 (`vit_base_patch16_dinov3.lvd1689m`), torchvision **MViTv2-S** (Kinetics-400), ResNet34 on kicker crops.
* No external datasets. No manual labelling, no use of private labels, and I did not search for the original broadcasts or any record of these penalties.

**LLM usage.** Done with Claude Code (Anthropic) in an agentic session. Chat history: *(attach the shareable link / PDF here before submitting, as the project's LLM policy requires)*.
"""))

cells.append(md(r"""
## 0. Setup and reproducibility notes
* `Q4data/train` = the 1,300 labelled *public* clips, `Q4data/val` = the 900 unlabelled *private* clips (folder names differ from the README).
* Every stage writes to `Q4_work/cache` and **skips work that is already cached**, so the notebook re-runs quickly; delete `cache/` (or set `FORCE_CNN=True`) to recompute from raw video. Full recomputation takes a few hours on one laptop GPU (RTX 5070 Ti): detection ≈ 30 min, pose ≈ 15 min, crops ≈ 8 min, keeper CNN ≈ 12 min. GPU training is not bit-for-bit deterministic, so a from-scratch re-run may change a few predictions.
* `RUN_OPTIONAL_EVAL=True` additionally re-runs the model-selection experiments (capacity sweep, keeper-CNN cross-validation, final out-of-fold evaluation), ≈ 30 min.
* Paths are Windows paths; edit `PROJECT_ROOT`, `YOLO_DET`, `YOLO_POSE` for another machine. `yolo11x.pt` and `yolo11m-pose.pt` are the standard Ultralytics release weights.
* Time handling: frames are chosen by **timestamp**, not index, because fps varies (15, 24, 25, 30, 50, 60) and the private set has 15/24 fps clips that training lacks.
"""))
cells.append(code(r"""
import os, sys, warnings
warnings.filterwarnings("ignore")
PROJECT_ROOT = r"C:\path\to\project-root"
YOLO_DET  = PROJECT_ROOT + r"\Q1-Q3work\yolo11x.pt"
YOLO_POSE = PROJECT_ROOT + r"\Q4_word_GPT\models\yolo11m-pose.pt"
FORCE_CNN = False          # retrain the keeper CNN even if its predictions are cached
RUN_OPTIONAL_EVAL = False  # re-run model-selection experiments (capacity sweep, CNN CV, final OOF evaluation)
os.makedirs(PROJECT_ROOT + r"\Q4_work\cache", exist_ok=True)
"""))

cells.append(md(r"""
## 1. Data observations that shaped the method
* Broadcast footage, mostly 1280x720; the kicker is only ~60-120 px tall, so full-frame embeddings are dominated by pitch and crowd. Players must be located and cropped.
* Camera angle varies. For about two-thirds of clips the goal runs almost horizontally across the image, so **image-left is not the kicker's left**. Everything is expressed in a **goal-facing frame**: forward = kicker→keeper direction in the image, lateral = forward rotated so that *positive = kicker's left*. In this frame a left/right mirror is exact (negate lateral axis, swap left/right joints, swap preferred foot, swap labels) which gives 2x data augmentation and test-time augmentation. The sign convention is validated empirically: keeper direction is predicted at 0.83+ from lateral keeper motion in this frame.
* 215 sources; 842 of 900 private clips come from sources never seen in training, so **all validation is grouped by `source_id`** (`StratifiedGroupKFold`, 5 folds, several seeds).
* `public_position_clicks.csv` (879 clips) is used only to *train and score* the kicker/keeper picker; no clicks exist or are used for private clips.
"""))

cells.append(md(r"""
## 2. Stage 1 - person detection at timed offsets (YOLO11x)
For every clip, 13 frames at fixed offsets before the last frame (1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.06, 0.03, 0 s) are run through YOLO11x (persons and balls), with a torso-colour descriptor per box.
"""))
cells.append(code(adapt(src("extract.py"))))

cells.append(md(r"""
## 3. Stage 2 - find the kicker and keeper
**Diagnosis that drove v3.** Breaking the v2 CV errors down showed that when the kicker box was wrong (13% of clips) kick accuracy was 0.42 vs 0.56 when right, and when the keeper box was wrong (17%) keeper accuracy was **0.54 vs 0.85**. Picking the right two people was therefore the most fixable loss.

* `track.py` (v2 picker): a gradient-boosted ranker scores each person box independently as kicker/keeper at the anchor frame (0.25 s before the end, where the clicks were made): 86.7% / 83.4%.
* `rank2.py` (**v3 picker**): scores **kicker-keeper pairs** jointly, adding (i) motion of each candidate over the surrounding frames (the kicker runs, referees and teammates stand), (ii) distance to a detected ball, (iii) pair geometry (the keeper stands a roughly fixed number of body-heights away) and (iv) colour difference between the pair. Out-of-fold by source: **kicker 96.4%, keeper 95.8%**. The ranker is trained on clicks only, and clips without clicks (and all private clips) use the model fit on all clicked clips.
Both players are then tracked across the 13 frames.
"""))
cells.append(code(adapt(src("track.py"))))
cells.append(code(adapt(src("rank2.py"))))

cells.append(md(r"""
## 4. Stage 3 - pose on upscaled crops, kinematic features
Each tracked player is cropped (1.6x box, 256 px) at every timed frame and run through YOLO11m-pose. 268 features are built in the goal-facing frame (`feats.py`): 9 joints relative to the hip centre at 0/0.06/0.15/0.25/0.4 s, hip/shoulder/knee line angles, trunk lean, ankle separation, kicker velocity, ball position, keeper centre displacement over time, keeper pose and body-line angles, relative geometry and metadata (preferred foot, gender, fps, duration). Missing joints are NaN (handled natively by the tree model).
"""))
cells.append(code(adapt(src("pose.py"))))
cells.append(code(adapt(src("feats.py"))))

cells.append(md(r"""
## 5. Stage 4 - source-grouped cross-validation of the tree models
`HistGradientBoostingClassifier` per target on original + mirrored samples; predictions are averaged with the mirrored-sample predictions. Results with the v3 picker (≈ ±0.014 standard error with 1,300 clips):

| features | kick | keeper | goal | score |
|---|---|---|---|---|
| majority class | 0.467 | 0.515 | 0.722 | 0.517 |
| preferred foot only | 0.502 | - | - | - |
| all features | 0.541 | 0.823 | 0.687 | 0.634 |
| kicker features only | **0.551** | 0.536 | 0.680 | 0.566 |
| keeper features only | 0.471 | **0.831** | 0.687 | 0.594 |

Decisions: **kick** = kicker static-pose block + preferred foot; **keeper** = all features; **outcome** = always `goal`.
"""))
cells.append(code(adapt(src("cv.py"))))

cells.append(md(r"""
## 6. Is the model overfitting? Capacity sweep and what the outcome can/can't do
**Train-vs-held-out gap.** Boosted trees reach 0.86-1.00 training accuracy but 0.55 (kick) / 0.83 (keeper) on held-out sources: a large *train* gap is normal for boosting. What matters is held-out accuracy, and it does **not** degrade as capacity increases, so I regularise by **bagging** (3 configs × several seeds, mirror TTA) rather than shrinking the trees. The sweep below (3 repeats of source-grouped 5-fold CV; selection among ~10 configs per target, so differences under ≈ 0.01 are noise):

| kick (91 feats) | CV | train | | keeper (268 feats) | CV | train |
|---|---|---|---|---|---|---|
| depth 2, 250 it | 0.542 | 0.72 | | depth 2 | 0.804 | 0.91 |
| depth 3, 100 it | 0.544 | 0.73 | | depth 3, 100 it | 0.815 | 0.91 |
| depth 3, 250 it (base) | 0.551 | 0.86 | | depth 3, 250 it (base) | 0.824 | 0.99 |
| depth 3, 500 it | 0.558 | 0.96 | | depth 3, 500 it | 0.830 | 1.00 |
| depth 4, 250 it | 0.561 | 0.95 | | depth 4, 250 it | 0.834 | 1.00 |
| logistic regression | 0.493 | 0.52 | | logistic regression | 0.688 | 0.72 |

The linear model sits at chance for kick, so the cue is nonlinear. The final models bag {depth 3/250, depth 4/250, depth 3/500} for kick and {depth 4/250, depth 3/500} for keeper.

**Keeper CNN.** Cross-validated ResNet34 on goal-aligned keeper crops (0 s and 0.25 s): 0.721 at 12 epochs, 0.851 at 20, **0.865 at 30** (3 frames: 0.860). It is still source-grouped, but the epoch count (12/20/30) was chosen on those folds. Blending with the tree model is flat between weights 0.4 and 0.8 (0.875-0.881), so I use 0.5 rather than the maximum.

**Goal outcome cannot be improved by a classifier.** When the keeper dives the same way as the kick 55% are goals; when the other way, 87%. Even in the favourable case `goal` is the majority class, so no prediction rule can beat always predicting `goal` (0.722).

**Things that did *not* help kick direction** (kick accuracy, grouped CV): frozen DINOv3 embeddings of kicker crops 0.46-0.50 (also on goal-aligned crops); ResNet34 fine-tuned on kicker crops 0.50 and 0.49 (also after the picker was fixed); ball position/motion features 0.537 vs 0.531; MViTv2-S (Kinetics-400) on 16 goal-aligned frames 0.514 / 0.510 on fold 0 (constant prediction) for wide and tight foot/ball crops. Private clips have **no near-duplicates** in the training set, and a hypothesis that the ball is in flight in the last frames was tested and rejected.
"""))
cells.append(code(wrap(selfcontained(src("tune.py")), "RUN_OPTIONAL_EVAL")))

cells.append(md(r"""
## 7. Stage 5 - keeper CNN (ResNet34 on goal-aligned keeper crops)
`align.py` rotates each crop so the kicker→keeper (goal) direction points up; then "kicker's left" is screen-left and a horizontal flip is an exact mirror. A shared ResNet34 processes the keeper crop at 0 s and 0.25 s, features are concatenated with preferred foot, trained with flip+label-swap augmentation, label smoothing, one-cycle LR, bf16. The next cells (optional) cross-validate it and evaluate the complete final configuration out of fold.
"""))
cells.append(code(adapt(src("align.py"))))
tc = adapt(src("train_cnn.py")).replace("arch, target, EP, folds = sys.argv[1], sys.argv[2], int(sys.argv[3]), [int(x) for x in sys.argv[4].split(\",\")]",
                                          "arch, target, EP, folds = 'resnet34.a1_in1k', 'gk', 30, [0, 1, 2, 3, 4]").replace("seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0", "seed = 0")
cells.append(code(wrap(tc, "RUN_OPTIONAL_EVAL")))
fe = selfcontained(src("final_eval.py")).replace('np.load(f"{C}/oof_gk_cnn30.npy")', 'np.load(f"{C}/oof_gk_resnet34.a1_in1k_0.npy")')
cells.append(code(wrap(fe, "RUN_OPTIONAL_EVAL")))

cells.append(md(r"""
## 8. Stage 6 - final training on all public clips, inference on private clips, CSV
* Keeper CNN: 5 seeds × 30 epochs on all 1,300 clips, flip-TTA predictions on the 900 private clips.
* Tree models: bagged configs × 3 seeds on all 1,300 clips (+ mirrors), mirror-TTA on private clips.
* Keeper = 0.5·trees + 0.5·CNN probabilities; kick = bagged trees (kicker pose + foot); outcome = `goal`.
The last cell writes `submission_v3.csv` and checks it against `submission_template.csv` (900 unique clips, exact class strings).
"""))
fc = src("final_cnn.py")
cells.append(code(wrap(adapt(fc), 'FORCE_CNN or not os.path.exists(PROJECT_ROOT + r"\\Q4_work\\cache\\cnn_gk_private.npy")')))
cells.append(code(adapt(src("make_sub3.py"))))

cells.append(md(r"""
## 9. Challenges, caveats and next steps
* **Kick direction remains the bottleneck** (60% of the score; 0.58 accuracy). The cue is nonlinear, mostly visible in favourable camera angles, and generic pretrained features or end-to-end fine-tuning on ~1,300 clips did not generalise to unseen sources. Short clips (< 1.4 s) are weaker (≈ 0.48) because there is little run-up.
* **What moved the score** was not a bigger model but fixing the *inputs*: the picker (+~0.03 over v2) and longer keeper-CNN training. Model-selection, checked by source-grouped CV, was kept to few configurations to limit selection bias; the blend weight and CNN epochs were still picked on the same out-of-fold predictions, so the ≈ 0.67 estimate is somewhat optimistic. The only independent check is the Kaggle leaderboard (v2: CV 0.62 vs public 0.61555).
* **Validation noise:** the 540 scored private clips give ≈ ±2 points standard error on kick accuracy.
* **Not done / not verified:** the video-model experiments were run on one fold; the picker has no ground truth for private clips (its held-out accuracy on training sources is the only evidence); no leaderboard score exists yet for v3.
"""))

nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, f"{R}\\Q4_Penalty_Kick.ipynb"); print("cells", len(cells))
