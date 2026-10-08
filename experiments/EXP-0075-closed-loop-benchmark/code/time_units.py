"""EXP-0075: timing of the basic units (model forward / forward+backward per candidate sequence, per model and horizon, batch-size sweep) -> results/timing_units.json / .md.
usage: PYTHONPATH=. python -u time_units.py"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments/EXP-0075-closed-loop-benchmark/code")); sys.path.insert(0, str(REPO / "experiments/EXP-0043-batched-closed-loop/code"))
from batched_closed_loop import goal_mask, goal_dist
from Genesis.binned_slate_dataset import BinnedSlateCorpus
import wide_planner as wp
from intensive import propose_from_occ, legal_first, rand_later, proj, R74
OUT = REPO / "experiments/EXP-0075-closed-loop-benchmark/results"
M = {
    "zoom128": lambda: [wp.Member("zoom", 128, R74 + "z128_f8_ms4/unet_best.pth")],
    "vanilla128": lambda: [wp.Member("world", 128, R74 + "w128_f8_ms4/unet_best.pth")],
    "ens128 (zoom+vanilla)": lambda: [wp.Member("zoom", 128, R74 + "z128_f8_ms4/unet_best.pth"), wp.Member("world", 128, R74 + "w128_f8_ms4/unet_best.pth")],
    "zoom64 (f8)": lambda: [wp.Member("zoom", 64, R74 + "z64_f8_ms4/unet_best.pth")],
    "vanilla64 (f4)": lambda: [wp.Member("world", 64, R74 + "w64_s0_ms4/unet_best.pth", feats=(4, 8, 16))],
    "vanilla64 (f8)": lambda: [wp.Member("world", 64, R74 + "w64_f8_full_ms4/unet_best.pth")],
}


def sync():
    torch.cuda.synchronize()


def main():
    rows = BinnedSlateCorpus.load(str(REPO / "Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys")).step(0); S0 = rows.states[(rows.slate_idx == 40).nonzero()[0, 0]].float(); goal = "letter_T_w20"; res = {}
    for name, mk in M.items():
        model = wp.WideEns(mk(), balance=True); obj = wp.SeqObjective(model, S0, goal_dist(goal), torch.from_numpy(goal_mask(goal)), 1.0, chunk=256); first = legal_first(propose_from_occ(model.occ0[0], 20000), S0); r = {}
        for H in (1, 2, 4):
            for chunk in (128, 256, 512):
                obj.chunk = chunk; N = 2048; seqs = torch.cat([first[torch.randint(0, len(first), (N,), device=wp.DEV)][:, None], rand_later(first, N, H)], 1) if H > 1 else first[torch.randint(0, len(first), (N,), device=wp.DEV)][:, None]
                try:
                    obj.cost(seqs[:256]); sync(); t0 = time.time(); obj.cost(seqs); sync(); r[f"fwd_H{H}_chunk{chunk}_ms_per_seq"] = (time.time() - t0) / N * 1000
                except torch.cuda.OutOfMemoryError:
                    r[f"fwd_H{H}_chunk{chunk}_ms_per_seq"] = None; torch.cuda.empty_cache()
        obj.chunk = 256; seqs = torch.cat([first[torch.randint(0, len(first), (24,), device=wp.DEV)][:, None], rand_later(first, 24, 4)], 1)
        for H in (1, 4):
            x = seqs[:, :H].clone().requires_grad_(True); opt = torch.optim.Adam([x], lr=2e-3)
            for it in range(5):
                if it == 2: sync(); t0 = time.time()
                opt.zero_grad(); xp = proj(x.reshape(-1, 4)).reshape(-1, H, 4); pr = model._predict(xp[..., :2], xp[..., 2:], grad=True)[-1]; obj._value(pr).sum().backward(); opt.step()
            sync(); r[f"gd_step_H{H}_ms_24seq"] = (time.time() - t0) / 3 * 1000
        res[name] = r; print(name, {k: (round(v, 3) if v else v) for k, v in r.items()}, flush=True)
        del model, obj; torch.cuda.empty_cache()
    json.dump(res, open(OUT / "timing_units.json", "w"), indent=1)


if __name__ == "__main__":
    main()
