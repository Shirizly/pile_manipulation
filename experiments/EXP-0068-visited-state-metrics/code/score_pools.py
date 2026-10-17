"""EXP-0068 scoring: every OCC model on a same-state pool corpus, with EXP-0059 eval_extended / EXP-0053 eval_narrow's
metric functions imported unchanged. Per ROW (goal-free): swept-region squared-error sums for accuracy_1 (hard-raster
truth, eval_narrow.acc's num/den), the same after Gaussian blur sigma 2 (eval_extended.blurred_acc), the sliced-W1
swept-region distance (eval_extended.sliced_w1_pair); per POOL x LETTER (lyapunov, soft truth occ_for_scoring): vt / vp
arrays (slateN capture, within-pool Spearman, optimism, top-1 regret follow from them) and |mass_in_goal_pred -
mass_in_goal_true| per row. Predicted occupancies persisted (float16) under artifacts/RUN-0003/.
Corpora: ds0023 (datasets/DS-0023-*/data/pools_0.pt, visited states) and ds0016 (DS-0016 test_pools_v2, spawn states).
Usage: python -u score_pools.py --corpus ds0023|ds0016 --models m1 m2 ...
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0059-retrieval-transition-model/code"))
sys.path.insert(0, str(REPO / "experiments/EXP-0053-narrow-domain-models/code"))
import eval_extended as EE  # noqa: E402
import eval_narrow as EN  # noqa: E402
from simple_mpc.adapters import make_occ_adapter, occ_for_scoring, occ_from_particles  # noqa: E402
from simple_mpc.learned_mpc import lyap  # noqa: E402
from Baselines.common.goals import mass_in_region  # noqa: E402

HERE = REPO / "experiments/EXP-0068-visited-state-metrics"
LETTERS = ["letter_O", "letter_T", "letter_S", "letter_L", "letter_X", "letter_Z", "letter_C", "letter_H"]
CORPORA = {"ds0023": REPO / "datasets/DS-0023-visited-state-pools/data/pools_0.pt",
           "ds0016": REPO / "Genesis/data/narrow_l20_n20/test_pools_v2/pools_0.pt"}


def _atomic_json(obj, path):
    tmp = str(path) + ".tmp"; Path(tmp).write_text(json.dumps(obj)); os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=list(CORPORA))
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_p = Path(a.out) if a.out else HERE / f"results/scores_{a.corpus}.json"
    art = HERE / "artifacts/RUN-0003"; art.mkdir(parents=True, exist_ok=True)
    out = json.loads(out_p.read_text()) if out_p.exists() else {}
    d = torch.load(CORPORA[a.corpus], map_location="cpu", weights_only=False)
    d = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in d.items()}
    R = len(d["states"]); act_all = torch.cat([d["p_starts"][:, :2], d["p_stops"][:, :2]], 1).float()
    pools = torch.unique(d["pool_idx"]).tolist()
    Dist = EE._dist_fields(LETTERS); Masks = EE._masks(LETTERS)
    print(f"{a.corpus}: {R} rows, {len(pools)} pools, device {dev}", flush=True)
    # corpus-level, model-free quantities (computed once)
    region = torch.cat([EN.swept_region(act_all[i:i + 256], "cpu").cpu() for i in range(0, R, 256)])
    truth_hard = occ_from_particles(d["states_"].float()).float().cpu()
    prev_hard = occ_from_particles(d["states"].float()).float().cpu()
    truth_soft = occ_for_scoring(d["states_"][:, :, :3].float()).float().cpu()
    prev_soft = occ_for_scoring(d["states"][:, :, :3].float()).float().cpu()
    reg_f = region.float()
    den = (((prev_hard - truth_hard) ** 2) * reg_f).sum((1, 2))
    den_b2 = (((EE.gaussian_blur_occ(prev_hard, 2) - EE.gaussian_blur_occ(truth_hard, 2)) ** 2) * reg_f).sum((1, 2))
    vt = {g: (lyap(truth_soft, Dist[g]) - lyap(prev_soft, Dist[g])) for g in LETTERS}
    mt = {g: mass_in_region(truth_soft, Masks[g]) for g in LETTERS}
    if "_corpus" not in out:
        out["_corpus"] = dict(path=str(CORPORA[a.corpus]), rows=R, pools=pools,
                              pool_idx=d["pool_idx"].tolist(), source=(d["source"].tolist() if "source" in d else [0] * R),
                              legal_ok=(d["legal_ok"].tolist() if "legal_ok" in d else [True] * R),
                              valid=d["valid"].tolist(), start_kind=list(d.get("start_kind", ["?"] * R)),
                              meta=d.get("meta"), den=den.tolist(), den_b2=den_b2.tolist(),
                              vt={g: vt[g].tolist() for g in LETTERS}, mass_true={g: mt[g].tolist() for g in LETTERS},
                              max_disp_mm=((d["states_"][:, :, :2] - d["states"][:, :, :2]).norm(dim=-1).max(1).values * 1000).tolist())
        _atomic_json(out, out_p)
    for m in a.models:
        if m in out:
            print("skip (scored):", m); continue
        t0 = time.time()
        try:
            ad = make_occ_adapter(m, dev, "corner")
        except (KeyError, NotImplementedError):
            print("skip (not registered):", m); continue
        preds = torch.zeros(R, 64, 64)
        for pi in pools:
            ix = torch.nonzero(d["pool_idx"] == pi)[:, 0]
            o0 = occ_from_particles(d["states"][ix[0]][None].float(), dev)
            o0_in = EE._encode(ad, o0)
            with torch.no_grad():
                for j in range(0, len(ix), 128):
                    jj = ix[j:j + 128]
                    preds[jj] = ad.predict_step(o0_in.expand(len(jj), -1, -1).contiguous(), act_all[jj].to(dev)).float().cpu()
        num = (((preds - truth_hard) ** 2) * reg_f).sum((1, 2))
        num_b2 = (((EE.gaussian_blur_occ(preds, 2) - EE.gaussian_blur_occ(truth_hard, 2)) ** 2) * reg_f).sum((1, 2))
        emd = torch.cat([EE.sliced_w1_pair(preds[i:i + 512] * reg_f[i:i + 512], truth_hard[i:i + 512] * reg_f[i:i + 512], dev)
                         for i in range(0, R, 512)])
        o0_all = prev_hard
        vp = {g: (lyap(preds, Dist[g]) - lyap(o0_all, Dist[g])).tolist() for g in LETTERS}
        mp = {g: (mass_in_region(preds, Masks[g]) - mt[g]).abs().tolist() for g in LETTERS}
        np.savez_compressed(art / f"pred_{a.corpus}_{m}.npz", pred=preds.numpy().astype(np.float16))
        out[m] = dict(num=num.tolist(), num_b2=num_b2.tolist(), emd=emd.tolist(), vp=vp, mass_abs_err=mp,
                      wall_s=time.time() - t0, device=dev)
        _atomic_json(out, out_p)
        acc1 = float(1 - torch.sqrt(num.sum() / den.sum()))
        print(f"{m:36s} accuracy_1 {acc1:+.3f} ({time.time() - t0:.0f}s)", flush=True)
        if dev == "cuda":
            del ad; torch.cuda.empty_cache()
    print("wrote", out_p)


if __name__ == "__main__":
    main()
