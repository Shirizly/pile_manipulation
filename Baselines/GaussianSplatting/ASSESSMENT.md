# Gaussian Splatting VMPC — assessment (agent C1-gs-assess)

**Verdict: SKIP.** No implementation attempted beyond this document.

**One-sentence reason.** On this dataset the paper's perception module
(multi-view RGBD → per-frame 3DGS reconstruction) is not just hard, it is
*moot* — we already have exact ground-truth Gaussians for free — and once
that shortcut is taken, the paper's actual dynamics model collapses onto the
same interaction-network family already implemented, trained and scored as
`Baselines/GNN` (accuracy 0.253/0.399), so there is nothing left to
reproduce that isn't a copy of an existing result.

Paper read in full: `docs/reference_papers/Gaussian Splatting Visual MPC for
Granular Media Manipulation.pdf` (7 pages, extracted with `pymupdf`, no page
images needed — `poppler-utils`/`pdftoppm` is not installed on this machine,
noted for future agents).

---

## 1. What the paper actually uses 3DGS for

Three roles, and it is important to keep them apart because only one of
them is a *dynamics model* (the only part our harness can score):

1. **State representation** (perception module `h`, §IV-B). Given
   multi-view RGBD + calibrated camera poses, they lift each view to a
   point cloud, FPS-downsample it, and run a **per-frame 3D Gaussian
   Splatting optimisation** (photometric fit to the multi-view images,
   Eq. 3-5, SSIM+L1) to obtain a set of Gaussians `Z_t = {(c_i, σ_i, R_i,
   g_i, s_i)}` per node `i`: RGB colour, opacity, rotation, 3-D centre,
   scale. This is done **once per frame**, independently, to build the
   training set `D_GS` from `D_RGBD`. There is no dynamics here — it is a
   per-timestep reconstruction problem, solved by an off-the-shelf
   differentiable rasteriser (`gsplat` / `diff-gaussian-rasterization`,
   the paper doesn't name the exact package but this is the standard 3DGS
   optimisation loop, i.e. `nerfstudio`/Inria's original code).

2. **Dynamics model** (`f`, §IV-C, Eq. 8-14 — this is the only part
   comparable to our baselines). It is a **PropNet/DPI-Net-style GNN**
   (same lineage as the Dyn-Res baseline already in this repo's
   `model/gnn_dyn.py`) built on a graph of the Gaussians (edges = L2
   distance below threshold ω, node features = `(c_i, σ_i, R_i, g_i,
   s_i)` plus the action `u_t`). It predicts, per node, only a
   **translation `Δg_i` and a rotation `Δr_i`**; colour, opacity and
   scale are carried over unchanged (Eq. 12: `ĉ,α̂,ŝ = c,α,s`, no
   channel for them in the loss or output). Trained with **Chamfer
   distance** between predicted and ground-truth next-frame Gaussian
   sets, with a position term plus a quaternion-alignment term weighted
   by `λ` (Eq. 14) — no photometric/rendering loss anywhere in dynamics
   training.

3. **Cost function** (§IV-D, Eq. 15-16). A soft density field `d(x) =
   Σ_i σ_i · exp(-(x-g_i)ᵀ Σ_i⁻¹ (x-g_i))` queried at a fixed set of
   points `P`, matched against the same field computed from the goal
   Gaussians via squared error. This is a KDE-smoothed occupancy field —
   structurally the same idea as this repo's own swept-region occupancy
   mask / goal-score-tensor already used by `EulerianAdapter.compute_reward`
   (`simple_mpc/adapters.py`) and the harness's `accuracy` metric. Not
   novel relative to what we already score with.

**So: 3DGS is the *representation*, not the dynamics model.** The dynamics
model is a plain particle GNN with a couple of extra (frozen) appearance
attributes riding along; the only thing distinguishing it algorithmically
from `Baselines/GNN` is that it also predicts rotation, and uses Chamfer
instead of masked MSE (Chamfer degenerates to the same thing when node
identity/correspondence is already known, which it is here — see §3).

## 2. What are its inputs

Multi-view (the paper's real rig: **4** calibrated RealSense RGBD cameras,
sim: PyBullet-rendered multi-view RGBD) + camera poses, per timestep, fed
through a per-frame 3DGS optimisation. **Our data has none of this.**
Verified directly:

- `Genesis/data/slates_multistep/n20_L20mm/*_data.pt` contains exactly
  `{states (128,20,7), states_ (128,20,7), p_starts (128,3), p_stops
  (128,3), angles (128,)}` — poses and push endpoints, **zero image
  tensors, zero camera data** anywhere in the file.
- `simple_mpc/adapters.py`/`simple_mpc/mpc.py` carry exactly **one**
  camera (`env.get_cam_params()` is singular, called once; `EulerianAdapter`
  and `GNNAdapter` both take one `cam_params`, not a list). There is no
  multi-view rig anywhere in this codebase's Genesis pipeline, real or
  simulated.

Any image used for a Gaussian-splatting pipeline here would have to be
**rendered by us from the known poses**, not "reconstructed" from real or
even simulated captures — there is nothing to reconstruct *from*.

## 3. The obvious shortcut, and whether it collapses to the existing GNN

We know, exactly: 20 cube centres+quaternions per state (`states[...,:3]`,
`states[...,3:7]`), a fixed cube edge (5 mm, identical for all 20 cubes,
all envs, all cells), and can pick an arbitrary constant colour/opacity.
So the perception module `h` is the **identity map**: `g_i` = cube centre,
`R_i` = cube quaternion (already in our data, untouched), `s_i` = the
constant half-edge, `(c_i, σ_i)` = arbitrary constants that carry zero
information (uniform across every node, every sample — they can never
contribute a training signal). No reconstruction, no multi-view capture, no
per-frame 3DGS optimisation, no differentiable rasteriser is needed or even
useful — we would be *fitting* something we already have in closed form,
which can only add reconstruction noise on top of ground truth, never
information.

Taking that shortcut, what's left of "the Gaussian Splatting dynamics
model" is: a PropNet-style GNN over 20 nodes, action entering as a per-node
field (their `u_t` into `f_enc`; ours is literally the same `s_delta` field
`Baselines/GNN` already computes via `compute_s_delta`/`_gen_s_delta`),
predicting a translation per node — **this is `Baselines/GNN` exactly**,
already trained and scored (0.253 accuracy L20mm / 0.399 L40mm, beating the
pooled linear operator on both cells, see `ORCHESTRATION_LOG.md`). The
frozen `(c,σ,s)` channels the paper's node features carry are constants
here and would be discarded by any reasonable implementation (a network
cannot learn from a channel with zero variance across the whole dataset).

**The one real difference is the rotation head.** The paper predicts `Δr_i`
per Gaussian via the Chamfer+quaternion loss (Eq. 14); our existing GNN
predicts *only* translation and explicitly reattaches the input frame's
quaternion unchanged at rasterisation time
(`Baselines/GNN/predictor.py`: *"Model has no orientation head... reattach
the INPUT frame's quaternion unchanged rather than inventing one"*). That
is a genuine, checkable gap — but it is a **one-paragraph ablation of the
already-existing GNN** (add a rotation MLP head + a quaternion loss term),
not a Gaussian-Splatting-specific contribution, and it doesn't touch
anything that makes the paper's method a *splatting* method (no rendering,
no photometric loss, no differentiable rasteriser, no appearance modelling
survives the shortcut). Framing it as "we implemented Gaussian Splatting
VMPC" would be a mischaracterisation; it belongs, if pursued at all, as a
follow-up ablation under `Baselines/GNN/`, not as a new baseline here.

**Plain statement, as requested: yes, it collapses to the same dynamics
model we already have, modulo one small, separable rotation-prediction
question that is orthogonal to what the paper calls "Gaussian Splatting."**

## 4. What would actually be reproducible, ranked by value/hour

| # | option | est. hours (shared RTX 4070 Laptop, 8 GB) | value | verdict |
|---|---|---|---|---|
| 1 | **Nothing** (this document) | 0 | — | **chosen** |
| 2 | Add a rotation head + Chamfer/quaternion loss to `Baselines/GNN` (not a GS baseline; an ablation of one already scored) | ~2–3 h (small MLP head, loss term, retrain, re-score both cells against the existing pipeline) | low-medium: tests one narrow question ("does predicting rotation improve occupancy accuracy for a rotating 5 mm cube swept sideways") that is plausible but not what this paper is about | legitimate small follow-up **elsewhere**, not here |
| 3 | Faithful pipeline: render synthetic RGB(D) from known poses via `EulerianAdapter._w2c`-style single-view projection, install `gsplat`, run per-frame 3DGS optimisation to *recover* the Gaussians we already have exactly, train the paper's GNN dynamics on the *reconstructed* (noisier) Gaussians, score with the density-field cost | ~8–12 h (install+debug a rasteriser we've never used in this repo, build a renderer, run thousands of per-frame optimisations, then redo all of #2) | none foreseeable: single camera can't test the paper's actual claim (multi-view reconstruction of occluded geometry), and optimising to recover known ground truth can only match or underperform using it directly — pure downside risk, no informative upside | not worth it |

Ranked: **(1) > (2, but relocate it) > (3)**. Given the night's stated
priorities (working models scored > papers recreated > performance, and
this task is explicitly item 4 of 4 with instruction to skip on a "doesn't
make sense" finding), the correct action is to stop here.

## 5. What is NOT reproducible, and why

- **No multi-view RGBD, ever.** Confirmed by direct inspection of every
  `*_data.pt` file's keys (poses only) and of `simple_mpc/adapters.py` /
  `simple_mpc/mpc.py` (`env.get_cam_params()` — one camera, called once,
  no list/loop over views anywhere in the Genesis MPC path).
- **No camera calibration rig / no real robot** — obviously out of scope
  for this offline-slate harness regardless (the register's whole `Wave C`
  framing already assumes attempt-and-skip is fine).
- **No differentiable Gaussian rasteriser installed.** Checked directly in
  the `pme` env: `pip list` has no `gsplat`, no `diff-gaussian-rasterization`.
  Also no `torch_geometric` (the paper's own stated implementation library,
  PyTorch-Geometric) — not a blocker by itself, since `model/gnn_dyn.py`
  is a hand-rolled message-passing implementation that needs no PyG, but it
  confirms nothing in this direction has been prepared or vendored here.
- **A false friend worth flagging for future agents**: `model/diff_mass_push.py`
  in this repo has functions named `differentiable_push_splat*` and a plot
  row labelled "Splatting" — these are **Eulerian mass-splatting** (bilinear
  deposition of density onto a grid, classic graphics sense of "splat"),
  completely unrelated to 3D Gaussian Splatting radiance fields. Do not
  mistake one for the other when grepping this codebase.

## Reading for the orchestrator

The most valuable output of this assessment is #3 above, stated plainly:
**a "Gaussian Splatting" dynamics model on ground-truth-known cube poses is
not a different model from the particle GNN we already scored — it is the
same interaction network with inert appearance channels attached, minus a
rotation head that is a separate, small, cheap experiment in its own right
if the orchestrator wants it, and does not belong under this name.** There
is no version of "try it on this data" that produces new information about
the paper's actual contribution (learning a 3-D representation from
multi-view images) — that is the one thing our data cannot exercise, by
construction.
