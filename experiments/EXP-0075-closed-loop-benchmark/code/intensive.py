"""EXP-0075 "intensive multistep optimization": plan H=4 pushes for ONE task (letter_T_w20, DS-0006 start 40) with the ensemble model and very large optimisation budgets, NO simulation.
Objective: terminal value (lyapunov - in-goal fraction after the 4 pushes) as in the benchmark. Candidate pushes come from an OCCUPANCY-AWARE proposal (start 10-30 mm before a random occupied pixel of the relevant occupancy -- the true start state for step 1,
the model's PREDICTED state for later steps -- aimed through it, 20-70 mm, kept only if the blade swath covers occupied pixels); step-1 pushes are made touchdown-legal against the true cubes.
Methods (all evaluated with the same terminal objective):
  ref     CEM exactly as in the benchmark (pool 256, pop 256, 4 iterations; 1280 sequence evaluations)
  random  best of 40,000 random sequences (step 1 occupancy-aware at the start state, later steps drawn like the benchmark: random rows of the start-state bank)
  cem10k  CEM, population 10,000, 20 iterations (200,000 sequence evaluations), elite 5 %
  greedy  step by step: 10,000 candidate pushes for step j given the model-predicted state after the best prefix (beam width 1)
  beam    same with beam width 8 x 1,250 candidates = 10,000 per step
  gd      Adam on the 4x4 action vector from the best 16 sequences of cem10k + beam, 150 steps (the model is differentiable)
Outputs: results/intensive.json, results/intensive_arrays.npz, figures/intensive_<method>.png, figures/intensive_summary.png.   usage: PYTHONPATH=. python -u intensive.py"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments/EXP-0075-closed-loop-benchmark/code")); sys.path.insert(0, str(REPO / "experiments/EXP-0043-batched-closed-loop/code"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from batched_closed_loop import goal_mask, goal_dist
from Genesis.binned_slate_dataset import BinnedSlateCorpus
from Genesis.action_sampling import legalize_pushes
from model.retrieval.frame import yaw_from_quat
from simple_mpc.adapters import OCC_BOUNDS
from simple_mpc.learned_mpc import project_push, XY_MARGIN
import wide_planner as wp
DEV = wp.DEV; R74 = "experiments/EXP-0074-wide-domain-zoom-nfd/runs/"; OUT = REPO / "experiments/EXP-0075-closed-loop-benchmark"
GOAL, START, H = "letter_T_w20", 40, 4
LO_, HI_ = OCC_BOUNDS["x_min"], OCC_BOUNDS["x_max"]; PITCH = (HI_ - LO_) / 63
torch.manual_seed(0); np.random.seed(0)
proj = lambda x: project_push(x, 0.02, 0.07)[0]


def propose_from_occ(occ, n, thr=0.3, min_hits=2):
    """occ (64,64) on DEV (dim0 = world x). Pile-aware candidates for THAT occupancy -> (m<=n, 4) [sx, sy, ex, ey] metres."""
    ij = (occ > thr).nonzero()
    if len(ij) == 0:
        return torch.zeros(0, 4, device=DEV)
    m = 4 * n; pick = ij[torch.randint(0, len(ij), (m,), device=DEV)].float(); pix = LO_ + pick * PITCH
    ang = torch.rand(m, device=DEV) * 2 * np.pi; d = torch.stack([torch.cos(ang), torch.sin(ang)], 1); L = 0.02 + 0.05 * torch.rand(m, 1, device=DEV)
    s = pix - d * (0.010 + 0.020 * torch.rand(m, 1, device=DEV)); a = proj(torch.cat([s, s + d * L], 1)); s, e = a[:, :2], a[:, 2:]
    dd = e - s; Ln = dd.norm(dim=1, keepdim=True).clamp_min(1e-6); u = dd / Ln; perp = torch.stack([-u[:, 1], u[:, 0]], 1)
    tt = torch.linspace(0, 1, 7, device=DEV)[None, :, None] * Ln[:, None, :]                 # (m,7,1)
    sw = torch.linspace(-0.02, 0.02, 9, device=DEV)
    P = s[:, None, None, :] + u[:, None, None, :] * tt[:, :, None, :] + perp[:, None, None, :] * sw.reshape(1, 1, 9, 1)             # (m,7,9,2)
    r = ((P[..., 0] - LO_) / PITCH).round().long().clamp(0, 63); c = ((P[..., 1] - LO_) / PITCH).round().long().clamp(0, 63)
    hits = (occ[r, c] > thr).sum((1, 2)); keep = hits >= min_hits
    return a[keep][:n]


def legal_first(acts, S0):
    xy = S0[:, :2][None].expand(len(acts), -1, -1); yaw = yaw_from_quat(S0[:, 3:7])[None].expand(len(acts), -1)
    a2, shift, ok = legalize_pushes(acts.cpu(), xy.cpu(), yaw.cpu(), 0.0025, 0.02, 0.001, box=(LO_ + XY_MARGIN, HI_ - XY_MARGIN))
    return a2[ok & (shift == 0)].to(DEV)


def rand_later(bank, n, Hh):
    return bank[torch.randint(0, len(bank), (n, Hh - 1), device=DEV)]


def describe(model, obj, seq):
    """per-step predicted value change (sequential), standalone prediction of each action from the START state, and predicted occupancies."""
    sq = seq.to(DEV)[None]; pr = model.predict(sq[..., :2], sq[..., 2:]); seqv = [float(obj._value(p)[0] - obj.v0) for p in pr]
    solo = [model.predict(sq[:, k:k + 1, :2], sq[:, k:k + 1, 2:])[0] for k in range(sq.shape[1])]; solov = [float(obj._value(p)[0] - obj.v0) for p in solo]
    return dict(seq_value=seqv, solo_value=solov, pred=torch.cat(pr).cpu().numpy(), solo=torch.cat(solo).cpu().numpy())


def main():
    rows = BinnedSlateCorpus.load(str(REPO / "Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys")).step(0)
    S0 = rows.states[(rows.slate_idx == START).nonzero()[0, 0]].float(); dist, mask = goal_dist(GOAL), torch.from_numpy(goal_mask(GOAL))
    model = wp.WideEns([wp.Member("zoom", 128, R74 + "z128_f8_ms4/unet_best.pth"), wp.Member("world", 128, R74 + "w128_f8_ms4/unet_best.pth")], balance=True)
    obj = wp.SeqObjective(model, S0, dist, mask, 1.0, chunk=256); occ0 = model.occ0[0]
    t0 = time.time(); first = propose_from_occ(occ0, 60000); first = legal_first(first, S0); print(f"proposals at the start state: {len(first)} legal pile-aware pushes ({time.time()-t0:.1f}s)", flush=True)
    res, arrays = {}, {}

    def record(name, seq, cost, t, n_ev):
        d = describe(model, obj, seq); res[name] = dict(cost=float(cost), time_s=t, n_evals=n_ev, seq=seq.tolist(), seq_value=d["seq_value"], solo_value=d["solo_value"])
        arrays[name + "_seq"] = seq.numpy(); arrays[name + "_pred"] = d["pred"]; arrays[name + "_solo"] = d["solo"]
        print(f"{name:7s} terminal cost {cost:+.4f} | per-push sequential value {np.round(d['seq_value'], 3).tolist()} | each push alone from the start {np.round(d['solo_value'], 3).tolist()} | {t:.0f}s, {n_ev} evals", flush=True)

    # ref: the benchmark's CEM
    torch.manual_seed(1); t0 = time.time(); tr_ref = []; o = wp.plan_seq(obj, first[:4096].cpu(), "cem", H, trace=tr_ref); record("ref", o["seq"], o["cost"], time.time() - t0, o["n_evals"]); traces = {"ref": tr_ref}
    # random search: 40k sequences
    t0 = time.time(); best, bc = None, 9; tr_rand = []
    for blk in range(16):
        seqs = torch.cat([first[torch.randint(0, len(first), (2500,), device=DEV)][:, None], rand_later(first, 2500, H)], 1); c = obj.cost(seqs); i = int(c.argmin())
        if float(c[i]) < bc: best, bc = seqs[i].cpu(), float(c[i])
        tr_rand.append((2500 * (blk + 1), bc))
    traces["random"] = tr_rand
    record("random", best, bc, time.time() - t0, 40000)
    # CEM 10k x 20
    torch.manual_seed(2); t0 = time.time(); pool = torch.cat([first[torch.randint(0, len(first), (10000,), device=DEV)][:, None], rand_later(first, 10000, H)], 1); cost = obj.cost(pool); n_el = 500
    best_i = int(cost.argmin()); best, bc = pool[best_i].clone(), float(cost[best_i]); el = pool[cost.argsort()[:n_el]]; mean, std = el.mean(0), el.std(0).clamp_min(1e-3); allt = [(float(cost.min()))]; keep = [(pool, cost)]
    for it in range(20):
        s = proj((mean + std * torch.randn(10000, H, 4, device=DEV)).reshape(-1, 4)).reshape(10000, H, 4); c = obj.cost(s); k = int(c.argmin())
        if float(c[k]) < bc: best, bc = s[k].clone(), float(c[k])
        el = s[c.argsort()[:n_el]]; mean, std = el.mean(0), el.std(0).clamp_min(1e-3); allt.append(bc); keep.append((s, c))
    res["cem10k_curve"] = allt; traces["cem10k"] = [(10000 * (i + 1), v) for i, v in enumerate(allt)]; record("cem10k", best.cpu(), bc, time.time() - t0, 10000 * 21)
    S_all = torch.cat([k[0] for k in keep]); C_all = torch.cat([k[1] for k in keep]); top = S_all[C_all.argsort()[:16]]

    def beam(B, N, name):
        t0 = time.time(); beams = torch.zeros(1, 0, 4, device=DEV); n_ev = 0; steps_log = []; traces[name + "_steps"] = steps_log
        for j in range(H):
            cands = []
            for b in range(len(beams)):
                if j == 0:
                    cj = first[torch.randint(0, len(first), (N,), device=DEV)]
                else:
                    pr = model.predict(beams[b:b + 1, :, :2], beams[b:b + 1, :, 2:])[-1][0]; cj = propose_from_occ(pr, N)
                    if len(cj) == 0:
                        cj = first[torch.randint(0, len(first), (N,), device=DEV)]
                cands.append(torch.cat([beams[b:b + 1].expand(len(cj), -1, -1), cj[:, None]], 1))
            seqs = torch.cat(cands); c = obj.cost(seqs); n_ev += len(seqs); beams = seqs[c.argsort()[:B]]
            print(f"  {name} step {j + 1}: best value after {j + 1} pushes {float(c.min()):+.4f} (candidates {len(seqs)})", flush=True); steps_log.append((n_ev, float(c.min()), j + 1))
        return beams, float(c.min()), time.time() - t0, n_ev
    torch.manual_seed(3); bm, bcst, tt, ne = beam(1, 10000, "greedy"); record("greedy", bm[0].cpu(), bcst, tt, ne)
    torch.manual_seed(4); bm, bcst, tt, ne = beam(8, 1250, "beam"); record("beam", bm[0].cpu(), bcst, tt, ne); beam_top = bm
    # gradient refinement of the best sequences
    t0 = time.time(); init = torch.cat([top, beam_top])[:24].clone(); x = init.clone().requires_grad_(True); opt = torch.optim.Adam([x], lr=2e-3)
    tr_gd = []
    for it in range(150):
        opt.zero_grad(); xp = proj(x.reshape(-1, 4)).reshape(-1, H, 4); pr = model._predict(xp[..., :2], xp[..., 2:], grad=True)[-1]; vv = obj._value(pr)
        with torch.no_grad(): tr_gd.append((len(init) * (it + 1), float(obj.cost(xp.detach()).min())))   # validated (no-grad, balanced) cost of the best of the batch
        loss = vv.sum(); loss.backward(); opt.step()
    traces["gd"] = tr_gd
    with torch.no_grad():
        xp = proj(x.detach().reshape(-1, 4)).reshape(-1, H, 4); c = obj.cost(xp); i = int(c.argmin())
    record("gd", xp[i].cpu(), float(c[i]), time.time() - t0, 150 * len(init))
    res["traces"] = traces; json.dump(res, open(OUT / "results/intensive.json", "w"), indent=1); np.savez_compressed(OUT / "results/intensive_arrays.npz", start=occ0.cpu().numpy(), **arrays)

    # ------------------------------------------------------------------ figures
    F = OUT / "figures"; F.mkdir(exist_ok=True); EXT = [LO_ * 1000, HI_ * 1000] * 2; gm = goal_mask(GOAL).astype(float); cols = ["tab:red", "tab:orange", "tab:green", "tab:purple"]

    def show(ax, img, vmax=1.0):
        ax.imshow(np.asarray(img, float).T, origin="lower", extent=EXT, cmap="gray_r", vmin=0, vmax=vmax, interpolation="nearest"); ax.set_xlim(LO_ * 1000, HI_ * 1000); ax.set_ylim(HI_ * 1000, LO_ * 1000); ax.set_xticks([]); ax.set_yticks([])
        ax.contour(np.linspace(LO_ * 1000, HI_ * 1000, 64), np.linspace(LO_ * 1000, HI_ * 1000, 64), gm.T, levels=[0.5], colors="tab:blue", linewidths=0.8)

    def arrow(ax, a, c, k=None):
        ax.annotate("", xy=(a[2] * 1000, a[3] * 1000), xytext=(a[0] * 1000, a[1] * 1000), arrowprops=dict(arrowstyle="->", color=c, lw=1.8)); ax.plot([a[0] * 1000], [a[1] * 1000], "o", ms=3, color=c)
        if k is not None: ax.text(a[0] * 1000, a[1] * 1000, str(k), color=c, fontsize=9, weight="bold")
    for name in ("ref", "random", "cem10k", "greedy", "beam", "gd"):
        r = res[name]; seq = np.array(r["seq"]); pr, so = arrays[name + "_pred"], arrays[name + "_solo"]
        fig, ax = plt.subplots(3, H + 1, figsize=(3 * (H + 1), 9.4)); show(ax[0, 0], occ0.cpu().numpy())
        for k in range(H): arrow(ax[0, 0], seq[k], cols[k], k + 1)
        ax[0, 0].set_title("start (model raster) + the 4 pushes", fontsize=9); ax[1, 0].axis("off"); ax[2, 0].axis("off")
        for k in range(H):
            show(ax[0, k + 1], pr[k]); arrow(ax[0, k + 1], seq[k], cols[k], k + 1); ax[0, k + 1].set_title(f"IN SEQUENCE: predicted after push {k + 1}\nvalue change since start {r['seq_value'][k]:+.3f}", fontsize=8)
            show(ax[1, k + 1], so[k]); arrow(ax[1, k + 1], seq[k], cols[k], k + 1); ax[1, k + 1].set_title(f"ALONE: push {k + 1} applied to the START state\nvalue change {r['solo_value'][k]:+.3f}", fontsize=8)
            d = (pr[k] - (pr[k - 1] if k else occ0.cpu().numpy())); show(ax[2, k + 1], np.abs(d), vmax=0.6); ax[2, k + 1].set_title(f"|change made by push {k + 1} in the sequence|\nincrement {r['seq_value'][k] - (r['seq_value'][k - 1] if k else 0):+.3f}", fontsize=8)
        fig.suptitle(f"{GOAL} start {START}: '{name}'  terminal predicted value change {r['cost']:+.3f}  ({r['n_evals']} sequence evaluations, {r['time_s']:.0f}s)", fontsize=10); fig.tight_layout(); fig.savefig(F / f"intensive_{name}.png", dpi=85); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4)); names = ["ref", "random", "cem10k", "greedy", "beam", "gd"]
    for n in names:
        ax[0].plot(range(1, H + 1), res[n]["seq_value"], "o-", label=f"{n} ({res[n]['cost']:+.3f})")
    ax[0].set_xlabel("push in the sequence"); ax[0].set_ylabel("predicted value change since start (lower = better)"); ax[0].legend(fontsize=8); ax[0].set_title("sequential predicted progress")
    w = 0.13
    for i, n in enumerate(names): ax[1].bar(np.arange(H) + i * w, np.diff([0] + res[n]["seq_value"]), w, label=n)
    ax[1].set_xticks(np.arange(H) + 0.3); ax[1].set_xticklabels([f"push {k + 1}" for k in range(H)]); ax[1].set_title("predicted value INCREMENT per push (more negative = more progress)"); fig.tight_layout(); fig.savefig(F / "intensive_summary.png", dpi=90)


if __name__ == "__main__":
    main()
