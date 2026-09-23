"""experiments/temp/binned-calibration/calibrate.py

Diagnostic (not a new experiment): is the near-zero slateN in desc-mlp's
slates_binned eval (experiments/temp/desc-mlp/RESULTS.md) a real property
of that corpus, or an artifact of the scoring path desc-mlp used?

Step 1 (the whole point): run desc-mlp's OWN scoring code, UNCHANGED
(predict_switched, desc_pointmass_value, compute_descriptors, com_world_pixel
/bilinear_sample -- all imported, not rewritten), on Genesis/data/
slates_multistep (step 0 only, corner goal, lyapunov value fn, MODEL-0002
switched-linear operator only), pooled over n20_L10mm+n20_L20mm+n20_L40mm
(n=60 slates) -- the exact (dataset, goal, value_fn, model) cell EXP-0006
published as pooled slateN(K=N) = +0.461 (sem 0.061), vs persistence -0.093
(sem 0.071) and random +0.120 (sem 0.052).

If this reproduces, the harness is sound and slates_binned's near-zero
scores are a real corpus property (verdict a). If it does not, the harness
used for slates_binned is defective (verdict b).

Step 2 (if a): characterise WHY slates_binned differs -- fixed K=32/K=128
references (comparable across pool sizes, unlike raw slateN which is not,
per METRICS.md) on both corpora, and the spread of dv_true (not just
frac(dv_true==0)).
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/alon/Code/pile_manipulation")
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/temp/dmdc-lenbins")
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/temp/desc-mlp")
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/temp/stage3-slaten")
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/temp/slaten-broad")

from control_utility_test import lyapunov, lyapunov_weights  # noqa: E402
from Baselines.common.data import load_cell  # noqa: E402
from Baselines.common.goals import slate_n_capture  # noqa: E402
from Genesis.binned_slate_dataset import BinnedSlateCorpus  # noqa: E402

# --- desc-mlp's OWN scoring code, imported unchanged ---
from eval_control import (  # noqa: E402
    compute_descriptors, predict_switched, load_switched, com_world_pixel,
)

DEVICE = "cpu"
K_FIXED_LIST = [32, 128]
N_RESAMPLE = 150

MULTISTEP = {
    "n20_L10mm": dict(
        eval_cfg="configs/dataset/genesis_slates_multistep_n20_L10mm_eval.yaml",
        manifest="Genesis/data/slates_multistep/n20_L10mm/manifest.json"),
    "n20_L20mm": dict(
        eval_cfg="configs/dataset/genesis_slates_multistep_n20_L20mm_eval.yaml",
        manifest="Genesis/data/slates_multistep/n20_L20mm/manifest.json"),
    "n20_L40mm": dict(
        eval_cfg="configs/dataset/genesis_slates_multistep_n20_L40mm_eval.yaml",
        manifest="Genesis/data/slates_multistep/n20_L40mm/manifest.json"),
}


def corner_goal(H, W):
    dw = lyapunov_weights((H, W), "corner", DEVICE)
    mask = np.zeros((H, W), dtype=bool)
    mask[: H // 2, : W // 2] = True
    return torch.from_numpy(mask), dw


def readout_switched(phi_pred, sp_d, ep_d, H, W):
    com = phi_pred[:, 3:5]
    mass_hat = phi_pred[:, 0:1][:, 0] * (H * W)
    wr, wc = com_world_pixel(com[:, 0], com[:, 1], sp_d, ep_d, H, W)
    return wr, wc, mass_hat


def score_pool_exact(v_true, v_pred, slate_ids, higher):
    caps = []
    for sid in slate_ids.unique().tolist():
        rows = (slate_ids == sid).nonzero(as_tuple=True)[0]
        c = slate_n_capture(v_pred[rows], v_true[rows], higher)
        if c == c:
            caps.append(c)
    return caps


def score_pool_kfixed(v_true, v_pred, slate_ids, higher, k, n_resample=N_RESAMPLE, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for sid in slate_ids.unique().tolist():
        rows = (slate_ids == sid).nonzero(as_tuple=True)[0]
        n = rows.numel()
        if n < k:
            continue
        vt, vp = v_true[rows], v_pred[rows]
        per = []
        for _ in range(n_resample):
            idx = torch.randperm(n, generator=g)[:k]
            c = slate_n_capture(vp[idx], vt[idx], higher)
            if c == c:
                per.append(c)
        if per:
            out.append(float(np.mean(per)))
    return out


def sem(x):
    x = np.asarray(x)
    if len(x) < 2:
        return float("nan")
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def load_multistep_step0(name, cfg):
    slates = load_cell(cfg["eval_cfg"], "train", manifest_path=cfg["manifest"], tag=name)
    step0 = slates.step_idx == 0
    states = slates.states[step0][:, :, :3].float()
    states_ = slates.states_[step0][:, :, :3].float()
    p_start = slates.p_start[step0][:, :2].float()
    p_stop = slates.p_stop[step0][:, :2].float()
    slate_idx = slates.slate_idx[step0].long()
    return states, states_, p_start, p_stop, slate_idx


def main():
    print("[1/3] loading MODEL-0002 switched-linear operator...")
    sw_ops, sw_edges, sw_slices = load_switched()

    print("[2/3] Step 1 -- reproduce slates_multistep corner/lyapunov, MODEL-0002 (K=N exact)...")
    all_caps = {"switched_linear": [], "persistence": [], "random": []}
    all_kfixed = {k: {"switched_linear": [], "persistence": []} for k in K_FIXED_LIST}
    dv_true_spread = {"multistep": []}
    offset = 0
    per_dataset = {}
    for name, cfg in MULTISTEP.items():
        states, states_, p_start, p_stop, slate_idx = load_multistep_step0(name, cfg)
        occ0, occ1, phi0, length_m, s_px, e_px = compute_descriptors(states, states_, p_start, p_stop)
        H, W = occ0.shape[-2:]
        mask, dw = corner_goal(H, W)
        v_true = lyapunov(occ1, dw)  # cost: lower better
        v_pre = lyapunov(occ0, dw)  # persistence prediction

        phi_sw = predict_switched(sw_ops, sw_edges, phi0, length_m)
        wr_sw, wc_sw, mass_sw = readout_switched(phi_sw, s_px, e_px, H, W)
        # lyapunov value fn: readout is direct field sample, no mass needed
        from eval_control import bilinear_sample
        v_sw = bilinear_sample(dw.cpu(), wr_sw, wc_sw)

        g_rand = torch.Generator().manual_seed(0)
        rand_score = torch.rand(occ0.shape[0], generator=g_rand) * 2 - 1
        # lyapunov is a COST (higher_is_better=False); random should have no
        # consistent sign preference either way -- match eval_control's convention
        v_rand = -rand_score

        sid = slate_idx + offset * 1000
        offset += 1

        caps_sw = score_pool_exact(v_true, v_sw, sid, higher=False)
        caps_pe = score_pool_exact(v_true, v_pre, sid, higher=False)
        caps_rd = score_pool_exact(v_true, v_rand, sid, higher=False)
        all_caps["switched_linear"] += caps_sw
        all_caps["persistence"] += caps_pe
        all_caps["random"] += caps_rd

        for k in K_FIXED_LIST:
            if k <= occ0.shape[0] // slate_idx.unique().numel():
                all_kfixed[k]["switched_linear"] += score_pool_kfixed(v_true, v_sw, sid, False, k)
                all_kfixed[k]["persistence"] += score_pool_kfixed(v_true, v_pre, sid, False, k)

        # dv_true spread (per slate, range of v_true across the pool)
        for s in slate_idx.unique().tolist():
            rows = (slate_idx == s).nonzero(as_tuple=True)[0]
            dv_true_spread["multistep"].append(float(v_true[rows].max() - v_true[rows].min()))

        per_dataset[name] = {
            "n_slates": int(slate_idx.unique().numel()), "n_rows": int(occ0.shape[0]),
            "pool_size": int(occ0.shape[0] // slate_idx.unique().numel()),
            "switched_linear_slateN": float(np.mean(caps_sw)) if caps_sw else float("nan"),
            "persistence_slateN": float(np.mean(caps_pe)) if caps_pe else float("nan"),
            "random_slateN": float(np.mean(caps_rd)) if caps_rd else float("nan"),
        }
        print(f"  {name}: n_slates={slate_idx.unique().numel()}, pool={occ0.shape[0]//slate_idx.unique().numel()}, "
              f"switched={per_dataset[name]['switched_linear_slateN']:.3f}")

    pooled = {
        "switched_linear": {"mean": float(np.mean(all_caps["switched_linear"])),
                             "sem": sem(all_caps["switched_linear"]), "n": len(all_caps["switched_linear"])},
        "persistence": {"mean": float(np.mean(all_caps["persistence"])),
                        "sem": sem(all_caps["persistence"]), "n": len(all_caps["persistence"])},
        "random": {"mean": float(np.mean(all_caps["random"])),
                   "sem": sem(all_caps["random"]), "n": len(all_caps["random"])},
        "published_EXP-0006": {"switched_linear": [0.461, 0.061], "persistence": [-0.093, 0.071],
                                "random": [0.120, 0.052], "n": 60},
    }
    print(f"  POOLED (n={pooled['switched_linear']['n']}): switched_linear="
          f"{pooled['switched_linear']['mean']:.3f} (sem {pooled['switched_linear']['sem']:.3f}) "
          f"vs published +0.461 (sem 0.061)")

    print("[3/3] Step 2 -- characterise slates_binned (K=32/K=128 fixed refs, dv_true spread)...")
    corpus = BinnedSlateCorpus.load("Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm")
    rows = corpus.step(0)
    states_b = rows.states[:, :, :3].float()
    states_b_ = rows.states_[:, :, :3].float()
    p_start_b = rows.p_starts[:, :2].float()
    p_stop_b = rows.p_stops[:, :2].float()
    slate_idx_b = rows.slate_idx.long()
    occ0_b, occ1_b, phi0_b, length_m_b, s_px_b, e_px_b = compute_descriptors(
        states_b, states_b_, p_start_b, p_stop_b)
    Hb, Wb = occ0_b.shape[-2:]
    mask_b, dw_b = corner_goal(Hb, Wb)
    v_true_b = lyapunov(occ1_b, dw_b)
    v_pre_b = lyapunov(occ0_b, dw_b)
    phi_sw_b = predict_switched(sw_ops, sw_edges, phi0_b, length_m_b)
    wr_b, wc_b, _ = readout_switched(phi_sw_b, s_px_b, e_px_b, Hb, Wb)
    from eval_control import bilinear_sample
    v_sw_b = bilinear_sample(dw_b.cpu(), wr_b, wc_b)

    binned_kfixed = {}
    for k in K_FIXED_LIST:
        binned_kfixed[k] = {
            "switched_linear": score_pool_kfixed(v_true_b, v_sw_b, slate_idx_b, False, k),
            "persistence": score_pool_kfixed(v_true_b, v_pre_b, slate_idx_b, False, k),
        }

    dv_spread_binned = []
    for s in slate_idx_b.unique().tolist():
        rows_i = (slate_idx_b == s).nonzero(as_tuple=True)[0]
        dv_spread_binned.append(float(v_true_b[rows_i].max() - v_true_b[rows_i].min()))

    def summarize_spread(x):
        x = np.asarray(x)
        return {"mean": float(x.mean()), "std": float(x.std()), "min": float(x.min()),
                "max": float(x.max()), "median": float(np.median(x))}

    results = {
        "step1_reproduction": {"per_dataset": per_dataset, "pooled": pooled},
        "step2_characterisation": {
            "multistep_dv_true_spread_per_slate": summarize_spread(dv_true_spread["multistep"]),
            "binned_dv_true_spread_per_slate": summarize_spread(dv_spread_binned),
            "binned_pool_size": int(occ0_b.shape[0] // slate_idx_b.unique().numel()),
            "multistep_pool_size": 128,
            "K_fixed_references": {
                "multistep": {str(k): {
                    "switched_linear_mean": float(np.mean(v["switched_linear"])) if v["switched_linear"] else None,
                    "persistence_mean": float(np.mean(v["persistence"])) if v["persistence"] else None,
                } for k, v in all_kfixed.items()},
                "binned": {str(k): {
                    "switched_linear_mean": float(np.mean(v["switched_linear"])) if v["switched_linear"] else None,
                    "persistence_mean": float(np.mean(v["persistence"])) if v["persistence"] else None,
                } for k, v in binned_kfixed.items()},
            },
        },
    }
    with open("/home/alon/Code/pile_manipulation/experiments/temp/binned-calibration/results_calibration.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("wrote results_calibration.json")


if __name__ == "__main__":
    main()
