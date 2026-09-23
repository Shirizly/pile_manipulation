"""EXP-0016 / RUN-0004: slateN on DS-0001 via the decoder readout, on the
same code path EXP-0014 used for its image-space models.

Reused verbatim from `scripts/probes/binned_pool_cache.py` (not reimplemented):
the corpus reader `BinnedSlateCorpus`, the rasteriser `particles_to_occupancy`
with the same BOUNDS/GRID/RADIUS, the value function `lyapunov` with
`lyapunov_weights`, the `dv = value(after) - value(before)` convention, and the
`random`/`persistence` baseline construction (same generator seed 0).  slateN
itself comes from the canonical `Baselines/common/goals.py::slate_n_capture`.

The model's dv_pred is: z0 = E(occ0) with the FROZEN RANDOM encoder,
z1_hat = z0 + F_K(z0, a), occ1_hat = D(z1_hat), dv = lyapunov(occ1_hat) -
lyapunov(occ0).  That is EXP-0014's image-space readout, with the predicted
image produced by this experiment's decoder.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from transforms.functional import particles_to_occupancy        # noqa: E402
from control_utility_test import lyapunov, lyapunov_weights     # noqa: E402
from Genesis.binned_slate_dataset import BinnedSlateCorpus      # noqa: E402
from Baselines.common.goals import slate_n_capture              # noqa: E402
from model import ResCNNEncoder, OccDecoder, SwitchedLinearDynamics, encode_action  # noqa: E402

BOUNDS = {"x_min": -0.064, "x_max": 0.064, "y_min": -0.064, "y_max": 0.064}
GRID = 64
RADIUS = 0.5 * 0.005 / ((BOUNDS["x_max"] - BOUNDS["x_min"]) / GRID)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def chunked(fn, n, size=512):
    return torch.cat([fn(lo, min(lo + size, n)) for lo in range(0, n, size)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(REPO / "Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm"))
    ap.add_argument("--enc", required=True)
    ap.add_argument("--dec", required=True)
    ap.add_argument("--dyn-dir", required=True)
    ap.add_argument("--ks", default="1,4,8")
    ap.add_argument("--dyn-seed", type=int, default=0)
    ap.add_argument("--goal", default="corner")
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    corpus = BinnedSlateCorpus.load(args.corpus)
    rows = corpus.step(args.step)
    n = len(rows)
    print(f"DS-0001: {corpus.n_slates} slates x {corpus.n_actions} candidates, "
          f"{corpus.spawn_style} spawn, {n} rows at step {args.step}", flush=True)

    states, states_ = rows.states.float(), rows.states_.float()
    ps, pe, ang = rows.p_starts[:, :2].float(), rows.p_stops[:, :2].float(), rows.angles.float()
    a = encode_action(ps, pe, ang).to(DEV)

    occ0 = chunked(lambda i, j: particles_to_occupancy(states[i:j, ..., :3].to(DEV), BOUNDS, (GRID, GRID), footprint_radius=RADIUS), n)
    occ1 = chunked(lambda i, j: particles_to_occupancy(states_[i:j, ..., :3].to(DEV), BOUNDS, (GRID, GRID), footprint_radius=RADIUS), n)
    assert occ0.device.type == DEV.split(":")[0], occ0.device

    dw = lyapunov_weights((GRID, GRID), args.goal, DEV)
    v_before = lyapunov(occ0, dw).cpu()
    dv_true = lyapunov(occ1, dw).cpu() - v_before

    ec = torch.load(args.enc, map_location=DEV)
    enc = ResCNNEncoder(**ec["config"]).to(DEV).eval()
    enc.load_state_dict(ec["state_dict"])
    dc = torch.load(args.dec, map_location=DEV)
    dec = OccDecoder(latent_dim=dc["latent_dim"], out_resolution=dc["out_resolution"]).to(DEV).eval()
    dec.load_state_dict(dc["state_dict"])
    mu, sd = dc["mu"].to(DEV), dc["sd"]

    with torch.no_grad():
        z0 = chunked(lambda i, j: enc(occ0[i:j, None]), n)
    print(f"z0 {tuple(z0.shape)} on {z0.device}", flush=True)

    preds = {"persistence": torch.zeros(n),
             "random": torch.rand(n, generator=torch.Generator().manual_seed(0))}
    # decoder-only floor: dv from D(z0), i.e. the readout with no dynamics at all
    with torch.no_grad():
        rec0 = chunked(lambda i, j: dec((z0[i:j] - mu) / sd), n)
    preds["decode-z0 (no dynamics)"] = (lyapunov(rec0, dw).cpu() - v_before)
    del rec0

    for K in [int(x) for x in args.ks.split(",")]:
        ck = torch.load(Path(args.dyn_dir) / f"dyn_K{K}_seed{args.dyn_seed}.pt", map_location=DEV)
        dyn = SwitchedLinearDynamics(**ck["config"]).to(DEV).eval()
        dyn.load_state_dict(ck["state_dict"])
        with torch.no_grad():
            def step(i, j):
                zp = dyn((z0[i:j] - mu) / sd, a[i:j])[1] + z0[i:j]
                return dec((zp - mu) / sd)
            occ_p = chunked(step, n)
        preds[f"random-enc K={K}"] = (lyapunov(occ_p, dw).cpu() - v_before)
        del occ_p

    ep = rows.slate_idx.long()
    res = {}
    for name, p in preds.items():
        caps = []
        for s in range(corpus.n_slates):
            m = ep == s
            c = slate_n_capture(p[m], dv_true[m], higher_is_better=False)
            if not np.isnan(c):
                caps.append(c)
        caps = np.asarray(caps)
        res[name] = {"slateN_mean": float(caps.mean()),
                     "slateN_sem": float(caps.std(ddof=1) / np.sqrt(caps.size)),
                     "n_slates": int(caps.size),
                     "frac_pred_exactly_zero": float((p == 0).float().mean()),
                     "n_distinct_pred": int(torch.unique(p).numel())}
        print(f"  {name:28s} slateN {res[name]['slateN_mean']:+.4f} "
              f"(sem {res[name]['slateN_sem']:.4f}) distinct {res[name]['n_distinct_pred']}",
              flush=True)

    res["_dv_true"] = {"mean": float(dv_true.mean()), "sd": float(dv_true.std()),
                       "frac_zero": float((dv_true == 0).float().mean()),
                       "frac_improving": float((dv_true < 0).float().mean())}
    res["_config"] = {"corpus": args.corpus, "goal": args.goal, "step": args.step,
                      "grid": GRID, "value_fn": "lyapunov", "dyn_seed": args.dyn_seed,
                      "encoder": "frozen random, seed 0",
                      "reference_rows_EXP0014": {"MODEL-0001": 0.7399, "MODEL-0003": 0.5163,
                                                 "MODEL-0002": -0.0904, "random": -0.1409,
                                                 "persistence": -0.0042}}
    print("dv_true:", res["_dv_true"], flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
