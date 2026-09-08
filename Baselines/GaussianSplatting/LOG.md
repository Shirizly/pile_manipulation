# Gaussian Splatting VMPC baseline — LOG

## SUMMARY (agent C1-gs-assess, assessment phase, 2026-09-08)

**Status: SKIPPED after assessment. No model implemented, no training run.**
See `ASSESSMENT.md` for the full writeup (paper read in full via `pymupdf`
text extraction — `poppler-utils` is not installed here, so the `Read` tool's
PDF-page-image path does not work on this machine; text extraction was
sufficient and is faster for a text-only paper).

**Verdict in one line:** 3D Gaussian Splatting in this paper is a *state
representation* (learned from multi-view RGBD via per-frame photometric
optimisation) that a plain PropNet-style GNN then predicts translation+
rotation over; our data gives every Gaussian's ground-truth parameters for
free (20 known 5 mm cubes, poses in `states`), which makes the perception
half vacuous and collapses the dynamics half onto `Baselines/GNN`
(already trained, scored 0.253/0.399 accuracy, beats the pooled linear
operator on both cells). The one real gap — the paper predicts rotation,
our GNN reattaches the input quaternion unchanged — is a small, separable
ablation of the *existing* GNN baseline, not a Gaussian-Splatting-specific
contribution, and was left for the orchestrator to route elsewhere rather
than implemented here.

Confirmed directly (not assumed) before concluding:
- No `gsplat` / `diff-gaussian-rasterization` / `torch_geometric` in the
  `pme` conda env (`/home/alon/anaconda3/envs/pme`).
- Every `Genesis/data/slates_multistep/*/*_data.pt` file's keys are exactly
  `states, states_, p_starts, p_stops, angles` — no image or camera data.
- `simple_mpc/adapters.py` / `simple_mpc/mpc.py` use exactly one camera
  (`env.get_cam_params()`, singular) throughout — no multi-view rig exists
  in this codebase to begin with.
- `Baselines/GNN/predictor.py` confirms the existing GNN has no rotation
  head (reattaches input quaternion unchanged at rasterisation) — this is
  the one substantive difference from the paper's dynamics model, and it is
  orthogonal to "Gaussian Splatting" (no rendering/appearance/rasteriser
  survives the ground-truth shortcut either way).

Budget used: ~20 tool calls (paper read, env checks, data/adapter/GNN code
inspection, this writeup). No `Baselines/GaussianSplatting/**` code beyond
these two markdown files.

## Running notes

- 2026-09-08: assessment only, per task gating. See `ASSESSMENT.md` §§1-5
  for the full reasoning chain. Nothing further planned unless the
  orchestrator asks for the rotation-head ablation (which should live under
  `Baselines/GNN/`, not here, per the finding above).
