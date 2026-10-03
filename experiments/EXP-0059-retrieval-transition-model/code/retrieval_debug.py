"""Visual debug figure for the retrieval-based transition model (EXP-0059,
coordinator request 2026-09-28): one multi-panel PNG/PDF per (query state,
push) that makes every step of the pipeline inspectable by eye --
retrieval (which bank rows come back and how far), the push-frame transform
(is the query's own configuration preserved, just rotated/translated), and
delta transfer (is the donor's displacement applied to the RIGHT query cube,
in the RIGHT frame).

Panel layout (see `plot_retrieval_debug`):
    1. GLOBAL world frame: the whole query state, tray, push arrow.
    2-6. QUERY's OWN push frame (`model.retrieval.frame`), identical axis
       limits/aspect across all five:
    2. query vs its 1-NN donor's pre-push state (+ Hungarian correspondence).
    3. query vs its k=5 neighbours' pre-push states (translucent, ranked).
    4. 1-NN transfer (+ arrows) vs the true next state (outline).
    5. each of the k=5 neighbours' OWN individually-transferred prediction.
    6. the actual COMBINED prediction (`predictor.predict_particles`) vs
       true next state, per-cube mm error + confidence annotations.

Reuses `model.retrieval.{frame,bank,distance,predictor}` verbatim -- no
transform/search/transfer math is reimplemented here, only drawing, so the
figure is faithful to what `eval_retrieval.py` actually scores. Per-cube mm
error here is INDEX-matched (query cube i's true vs predicted position;
`predict_particles` never permutes query cube identity, only borrows a
Hungarian-matched donor's displacement for it) -- a different, more precise
quantity than the register's own "moved-cube mm error" (nearest-match,
correspondence-free, see `neighbour_rank_curve.py`/`diagnostics_r2.py`),
which exists for a different purpose (bounding oracle donors it hasn't
Hungarian-matched at all). Do not read the two numbers as the same metric.

Usage:
    python -u experiments/EXP-0059-retrieval-transition-model/code/retrieval_debug.py \\
        --bank experiments/EXP-0059-retrieval-transition-model/artifacts/bank.pt \\
        --n 12
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/probes"))
from cube_viz import cube_patch, push_arrow  # noqa: E402

from model.retrieval.bank import TransitionBank, OCC_BOUNDS, DEFAULT_MOVED_THRESHOLD  # noqa: E402
from model.retrieval.frame import (world_to_push_frame, push_frame_to_world, push_angle,
                                   yaw_from_quat, wrap_angle, tray_corners_push_frame)  # noqa: E402
from model.retrieval.distance import DistanceConfig, CUBE_SIZE, BLADE_HALF_WIDTH  # noqa: E402
from model.retrieval.predictor import RetrievalPredictor  # noqa: E402

D = REPO / "Genesis/data/narrow_l20_n20"
ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
FIGDIR = Path(__file__).resolve().parents[1] / "figures/retrieval_debug"

NEIGH_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
QUERY_COLOR = "#222222"
TRUE_COLOR = "#2ca02c"
PRED_COLOR = "#d62728"


# --------------------------------------------------------------------------
# data loading (DS-0016 test_chains_v2_clean -- ISS-010-fix sampler; DS-0009's
# own `test_chains` was collected with the pre-fix `_pile_aware_stops` clamp,
# which put the tool on top of a cube at touchdown in ~46% of rows -- see
# ISS-010 in experiments/OPEN_ISSUES.md. Querying against that set made this
# figure show the illegal-touchdown artifact it exists to catch, rather than
# a representative query. DS-0016 is the same shape/physics/seeds-aside
# replacement, collected with the fixed `pile_aware_action_batch` sampler,
# already split so bad rows never reach this script.)
# --------------------------------------------------------------------------
def _cpu(d):
    return {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in d.items()}


def load_test_chains():
    files = sorted(glob.glob(str(D / "test_chains_v2_clean/_*_data.pt")))
    ch = [_cpu(torch.load(f, map_location="cpu", weights_only=False)) for f in files]
    states0 = torch.cat([d["states"] for d in ch]).float()
    states1 = torch.cat([d["states_"] for d in ch]).float()
    p_starts = torch.cat([d["p_starts"] for d in ch]).float()
    p_stops = torch.cat([d["p_stops"] for d in ch]).float()
    valid = torch.cat([d["valid"] for d in ch])
    kinds = np.array(sum([list(d["start_kind"]) for d in ch], []))
    return states0, states1, p_starts, p_stops, valid, kinds


# --------------------------------------------------------------------------
# core drawing helpers
# --------------------------------------------------------------------------
def _draw_cubes(ax, xy, yaw, color, alpha=1.0, size=CUBE_SIZE, zorder=2, label=None):
    for i in range(xy.shape[0]):
        ax.add_patch(cube_patch(xy[i], float(yaw[i]), size, facecolor=color, alpha=alpha,
                                edgecolor="none", zorder=zorder))
    if label is not None:
        ax.plot([], [], marker="s", ls="none", color=color, alpha=max(alpha, 0.5), label=label)


def _draw_cubes_outline(ax, xy, yaw, color, size=CUBE_SIZE, lw=1.2, zorder=3, label=None):
    for i in range(xy.shape[0]):
        p = cube_patch(xy[i], float(yaw[i]), size, facecolor="none", edgecolor=color, lw=lw, zorder=zorder)
        ax.add_patch(p)
    if label is not None:
        ax.plot([], [], marker="s", ls="none", mfc="none", mec=color, label=label)


def _push_frame_axes(ax, window_u, window_v, corridor_u_hi, corridor_v_halfwidth,
                     push_len, tray_uv):
    """Draw the frame furniture shared by panels 2-6: retrieval window
    (dashed outline), swept corridor (faint fill), plate at start/end
    (thick lines at u=0 and u=push_len), tray walls (polygon)."""
    win = Rectangle((window_u[0], window_v[0]), window_u[1] - window_u[0], window_v[1] - window_v[0],
                    fill=False, ls="--", ec="#888888", lw=0.9, zorder=1)
    ax.add_patch(win)
    corridor = Rectangle((0.0, -corridor_v_halfwidth), corridor_u_hi, 2 * corridor_v_halfwidth,
                         facecolor="#ffd54f", alpha=0.18, edgecolor="none", zorder=0)
    ax.add_patch(corridor)
    ax.plot([0, 0], [-BLADE_HALF_WIDTH, BLADE_HALF_WIDTH], color="royalblue", lw=3, zorder=4)
    ax.plot([push_len, push_len], [-BLADE_HALF_WIDTH, BLADE_HALF_WIDTH], color="royalblue", lw=3,
            alpha=0.5, zorder=4, ls="--")
    push_arrow(ax, (0.0, 0.0), (push_len, 0.0), color="royalblue", lw=1.3)
    # tray corners come back as an UNORDERED (x_min/x_max) x (y_min/y_max)
    # cross product ([[x,y] for x in xs for y in ys], frame.py), not a
    # cyclic polygon order -- reorder [0,1,3,2] for a closed rectangle loop.
    order = [0, 1, 3, 2]
    poly = Polygon(tray_uv[order].numpy(), closed=True, fill=False, ec="#555555", lw=0.8,
                   ls=":", zorder=1)
    ax.add_patch(poly)


def _hungarian(uv_a: torch.Tensor, uv_b: torch.Tensor):
    cost = torch.cdist(uv_a, uv_b).numpy()
    return linear_sum_assignment(cost)


# --------------------------------------------------------------------------
# main figure
# --------------------------------------------------------------------------
def plot_retrieval_debug(query_state: torch.Tensor, action: dict, predictor: RetrievalPredictor,
                         bank: TransitionBank, true_next: torch.Tensor = None,
                         out_path: str = None, query_id=None, start_kind: str = None,
                         moved_threshold: float = DEFAULT_MOVED_THRESHOLD):
    """query_state: (20, 7) world state [x,y,z,qw,qx,qy,qz]. action: dict with
    `p_start`/`p_stop`, each (3,) world metres. `predictor`: a fitted
    `RetrievalPredictor` (k=5 aggregation="cube_median" is the intended
    subject). `bank`: the `TransitionBank` the predictor searches. `true_next`:
    optional (20,7) ground truth (DS-0009 `states_`). Writes `out_path` (.png)
    plus a vector twin (.pdf) alongside it if `out_path` ends in .png."""
    n = query_state.shape[0]
    p_start, p_stop = action["p_start"], action["p_stop"]
    p_start_b, p_stop_b = p_start[None].float(), p_stop[None].float()
    states0_b = query_state[None].float()

    # ---- retrieval search + combined prediction (exactly what eval_retrieval.py runs) ----
    idx, dist, uv0_q_b, q_valid, q_in_set_b = predictor._search(states0_b, p_start_b, p_stop_b)
    neighbours = idx[0]           # (k,)
    neighb_dist = dist[0]         # (k,)
    uv0_q = uv0_q_b[0]            # (n,2), query cubes in ITS OWN push frame
    q_in_set = q_in_set_b[0]      # (n,) bool -- the query's own geometric interaction set
    combined_pred, conf = predictor.predict_particles_with_confidence(states0_b, p_start_b, p_stop_b)
    combined_pred = combined_pred[0]
    top1_dist = float(conf["top1_dist"][0])
    knn_disagreement = float(conf["knn_disagreement"][0])

    yaw0_w = yaw_from_quat(query_state[:, 3:7])
    phi = float(push_angle(p_start_b, p_stop_b)[0])
    yaw0_pf = wrap_angle(yaw0_w - phi)                        # push-frame yaw, for drawing
    push_len = float((p_stop[:2] - p_start[:2]).norm())

    # 1-NN transfer (panel 4, ALWAYS rank-0, regardless of predictor.k/aggregation)
    duv_1nn, dyaw_1nn = predictor._transfer_one(uv0_q, int(neighbours[0]), q_in_set)
    uv1_1nn = uv0_q + duv_1nn
    yaw1_1nn = wrap_angle(yaw0_pf + dyaw_1nn) if predictor.transfer_yaw else yaw0_pf

    # each neighbour's own individually-transferred prediction (panel 5)
    per_neigh_uv1 = []
    for j in neighbours.tolist():
        duv_j, _ = predictor._transfer_one(uv0_q, int(j), q_in_set)
        per_neigh_uv1.append(uv0_q + duv_j)

    # combined prediction & true next, in the QUERY's OWN push frame
    uv1_combined = world_to_push_frame(combined_pred[None, :, :2], p_start_b, p_stop_b)[0]
    yaw1_combined = wrap_angle(yaw_from_quat(combined_pred[:, 3:7]) - phi)
    if true_next is not None:
        uv1_true = world_to_push_frame(true_next[None, :, :2], p_start_b, p_stop_b)[0]
        yaw1_true = wrap_angle(yaw_from_quat(true_next[:, 3:7]) - phi)
        true_moved = (true_next[:, :2] - query_state[:, :2]).norm(dim=-1) > moved_threshold
        per_cube_err_mm = (true_next[:, :2] - combined_pred[:, :2]).norm(dim=-1) * 1000.0
        moved_mm = float(per_cube_err_mm[true_moved].mean()) if bool(true_moved.any()) else float("nan")
    else:
        uv1_true = yaw1_true = true_moved = per_cube_err_mm = None
        moved_mm = float("nan")

    cfg = predictor.cfg
    tray_uv = tray_corners_push_frame(p_start_b, p_stop_b, OCC_BOUNDS)[0]
    corridor_u_hi = push_len + cfg.corridor_u_pad

    # shared axis limits for panels 2-6: cover window + a small margin
    ulo = min(cfg.window_u[0], -0.01) - 0.005
    uhi = max(cfg.window_u[1], push_len + 0.01) + 0.005
    vlo, vhi = cfg.window_v[0] - 0.005, cfg.window_v[1] + 0.005

    fig, axes = plt.subplots(2, 3, figsize=(21, 13))
    ((ax1, ax2, ax3), (ax4, ax5, ax6)) = axes

    # ---- panel 1: GLOBAL world frame ----
    tray = Rectangle((OCC_BOUNDS["x_min"], OCC_BOUNDS["y_min"]),
                     OCC_BOUNDS["x_max"] - OCC_BOUNDS["x_min"],
                     OCC_BOUNDS["y_max"] - OCC_BOUNDS["y_min"],
                     fill=False, ec="#555555", lw=1.0, ls=":")
    ax1.add_patch(tray)
    _draw_cubes(ax1, query_state[:, :2], yaw0_w, QUERY_COLOR, alpha=0.85)
    push_arrow(ax1, p_start[:2].tolist(), p_stop[:2].tolist(), color="royalblue", lw=1.6)
    m = 0.015
    ax1.set_xlim(OCC_BOUNDS["x_min"] - m, OCC_BOUNDS["x_max"] + m)
    ax1.set_ylim(OCC_BOUNDS["y_min"] - m, OCC_BOUNDS["y_max"] + m)
    ax1.set_aspect("equal"); ax1.set_title("1. GLOBAL world frame: query state + push", fontsize=10)
    ax1.set_xlabel("world x [m]"); ax1.set_ylabel("world y [m]")

    for ax, title in ((ax2, "2. query vs 1-NN donor (pre-push) + correspondence"),
                      (ax3, "3. query vs k=5 neighbours (pre-push)"),
                      (ax4, "4. 1-NN transfer vs TRUE next (outline)"),
                      (ax5, "5. each neighbour's own transferred prediction"),
                      (ax6, "6. COMBINED prediction (used) vs TRUE next")):
        _push_frame_axes(ax, cfg.window_u, cfg.window_v, corridor_u_hi, cfg.corridor_v_halfwidth,
                         push_len, tray_uv)
        ax.set_xlim(ulo, uhi); ax.set_ylim(vlo, vhi); ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("push-frame u [m]"); ax.set_ylabel("push-frame v [m]")

    # ---- panel 2: query vs 1-NN donor + Hungarian correspondence ----
    donor0_uv0 = bank.uv0[int(neighbours[0])]
    donor0_yaw0 = bank.yaw0[int(neighbours[0])]
    _draw_cubes(ax2, uv0_q, yaw0_pf, QUERY_COLOR, alpha=0.9, label="query (pre-push)")
    _draw_cubes(ax2, donor0_uv0, donor0_yaw0, NEIGH_COLORS[0], alpha=0.45, label="1-NN donor (pre-push)")
    row, col = _hungarian(uv0_q, donor0_uv0)
    for r, c in zip(row, col):
        ax2.plot([uv0_q[r, 0], donor0_uv0[c, 0]], [uv0_q[r, 1], donor0_uv0[c, 1]],
                color="#999999", lw=0.6, zorder=1)
    ax2.legend(fontsize=7, loc="upper right")

    # ---- panel 3: query vs k=5 neighbours ----
    _draw_cubes(ax3, uv0_q, yaw0_pf, QUERY_COLOR, alpha=0.9, label="query (pre-push)")
    for r in range(len(neighbours)):
        j = int(neighbours[r])
        c = NEIGH_COLORS[r % len(NEIGH_COLORS)]
        _draw_cubes(ax3, bank.uv0[j], bank.yaw0[j], c, alpha=0.3,
                   label=f"rank {r + 1}, d={float(neighb_dist[r]):.4f}")
    ax3.legend(fontsize=6.5, loc="upper right")

    # ---- panel 4: 1-NN transfer + arrows vs true next ----
    _draw_cubes(ax4, uv0_q, yaw0_pf, "#bbbbbb", alpha=0.5, label="query (pre-push)")
    _draw_cubes(ax4, uv1_1nn, yaw1_1nn, PRED_COLOR, alpha=0.85, label="1-NN transferred prediction")
    for i in range(n):
        if float(duv_1nn[i].norm()) > moved_threshold:
            push_arrow(ax4, uv0_q[i].tolist(), uv1_1nn[i].tolist(), color=PRED_COLOR, lw=0.9)
    if uv1_true is not None:
        _draw_cubes_outline(ax4, uv1_true, yaw1_true, TRUE_COLOR, lw=1.3, label="true next state")
    ax4.legend(fontsize=7, loc="upper right")

    # ---- panel 5: each neighbour's own transferred prediction ----
    _draw_cubes(ax5, uv0_q, yaw0_pf, "#bbbbbb", alpha=0.4, label="query (pre-push)")
    for r, uv1_r in enumerate(per_neigh_uv1):
        c = NEIGH_COLORS[r % len(NEIGH_COLORS)]
        _draw_cubes(ax5, uv1_r, yaw0_pf, c, alpha=0.3, label=f"rank {r + 1} transferred pred.")
    if uv1_true is not None:
        _draw_cubes_outline(ax5, uv1_true, yaw1_true, TRUE_COLOR, lw=1.1, label="true next state")
    ax5.legend(fontsize=6.5, loc="upper right")

    # ---- panel 6: combined prediction vs true next, per-cube error ----
    _draw_cubes(ax6, uv1_combined, yaw1_combined, PRED_COLOR, alpha=0.8,
               label=f"combined pred. ({predictor.name})")
    if uv1_true is not None:
        _draw_cubes_outline(ax6, uv1_true, yaw1_true, TRUE_COLOR, lw=1.3, label="true next state")
        for i in range(n):
            if bool(true_moved[i]) or float(per_cube_err_mm[i]) > 1.0:
                ax6.annotate(f"{float(per_cube_err_mm[i]):.1f}", uv1_combined[i].tolist(),
                            fontsize=6, color="#333333", ha="center", va="bottom")
    ax6.legend(fontsize=7, loc="upper right")
    conf_txt = (f"top1_dist={top1_dist:.4f}  knn_disagreement={knn_disagreement:.4f}\n"
               f"moved-cube mean |err| (index-matched) = {moved_mm:.2f} mm")
    ax6.text(0.02, 0.02, conf_txt, transform=ax6.transAxes, fontsize=7.5,
            va="bottom", ha="left", bbox=dict(fc="white", alpha=0.75, ec="#999999", lw=0.5))

    kind_str = f", {start_kind}" if start_kind else ""
    fig.suptitle(f"retrieval debug -- query {query_id}{kind_str}  |  predictor={predictor.name}  |  "
                f"moved-cube mean |err| = {moved_mm:.2f} mm  |  L={push_len * 1000:.0f} mm", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=220)
        fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return dict(moved_mm=moved_mm, top1_dist=top1_dist, knn_disagreement=knn_disagreement)


# --------------------------------------------------------------------------
# batch CLI: pick ~12 informative queries and render all of them
# --------------------------------------------------------------------------
def _wall_distance_world(xy: torch.Tensor, bounds=OCC_BOUNDS) -> torch.Tensor:
    dx0 = xy[..., 0] - bounds["x_min"]; dx1 = bounds["x_max"] - xy[..., 0]
    dy0 = xy[..., 1] - bounds["y_min"]; dy1 = bounds["y_max"] - xy[..., 1]
    return torch.minimum(torch.minimum(dx0, dx1), torch.minimum(dy0, dy1))


def select_queries(states0, states1, p_starts, p_stops, kinds, predictor, n_total=12):
    """Batched confidence/error pass over every valid DS-0016 test_chains row,
    then a stratified pick: typical scatter/clump, near-wall pushes, best/
    worst by moved-cube mm error, highest retrieval disagreement."""
    N = states0.shape[0]
    _, conf = predictor.predict_particles_with_confidence(states0, p_starts, p_stops)
    pred = predictor.predict_particles(states0, p_starts, p_stops)
    true_moved = (states1[:, :, :2] - states0[:, :, :2]).norm(dim=-1) > DEFAULT_MOVED_THRESHOLD
    err_mm = (states1[:, :, :2] - pred[:, :, :2]).norm(dim=-1) * 1000.0
    moved_mm = torch.tensor([float(err_mm[i][true_moved[i]].mean()) if bool(true_moved[i].any())
                            else float("nan") for i in range(N)])
    wall_mm = _wall_distance_world(p_starts[:, :2]) * 1000.0
    knn_dis = conf["knn_disagreement"]

    picked, seen = [], set()

    def take(order, k, tag):
        c = 0
        for i in order:
            i = int(i)
            if i in seen:
                continue
            seen.add(i); picked.append((i, tag)); c += 1
            if c >= k:
                break

    rng = np.random.default_rng(0)
    scatter_ix = np.nonzero(kinds == "scatter")[0]
    clump_ix = np.nonzero(kinds == "clump")[0]
    take(rng.permutation(scatter_ix), 2, "scatter_typical")
    take(rng.permutation(clump_ix), 2, "clump_typical")
    take(torch.argsort(wall_mm), 2, "near_wall")
    finite = torch.nonzero(torch.isfinite(moved_mm) & (moved_mm > 0.5))[:, 0]
    take(finite[torch.argsort(moved_mm[finite])], 2, "best_mm")
    take(finite[torch.argsort(-moved_mm[finite])], 2, "worst_mm")
    take(torch.argsort(-knn_dis), 2, "high_disagreement")
    # top up to n_total if any category ran dry (dedup collisions)
    if len(picked) < n_total:
        rest = [i for i in range(N) if i not in seen]
        rng.shuffle(rest)
        for i in rest[: n_total - len(picked)]:
            seen.add(i); picked.append((i, "fill"))
    return picked[:n_total], dict(moved_mm=moved_mm, wall_mm=wall_mm, knn_disagreement=knn_dis, err_mm=err_mm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=str(ARTIFACTS / "bank.pt"))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default=str(FIGDIR))
    a = ap.parse_args()

    bank = TransitionBank.load(a.bank)
    print(f"loaded bank: {len(bank)} transitions from {a.bank}")
    predictor = RetrievalPredictor(
        bank, cfg=DistanceConfig(cap=0.02, mismatch_penalty=0.04, corridor_weight=3.0),
        k=5, aggregation="cube_median")

    states0, states1, p_starts, p_stops, valid, kinds = load_test_chains()
    v = valid.bool()
    states0, states1, p_starts, p_stops, kinds = states0[v], states1[v], p_starts[v], p_stops[v], kinds[v.numpy()]
    print(f"DS-0016 test_chains_v2_clean: {states0.shape[0]} valid rows")

    picked, stats = select_queries(states0, states1, p_starts, p_stops, kinds, predictor, n_total=a.n)

    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    bank_rel = os.path.relpath(a.bank, REPO)
    index_lines = ["# Retrieval debug figures (EXP-0059)", "",
                  f"Bank: `{bank_rel}` ({len(bank)} transitions). Predictor: `{predictor.name}` "
                  f"(k=5, cube_median, cap=0.02, mismatch_penalty=0.04, corridor_weight=3.0). "
                  "Query source: DS-0016 `test_chains_v2_clean` (ISS-010-fix sampler; valid "
                  "rows only).", "",
                  "| file | query idx | start kind | why picked | moved-mm | top1_dist | knn_disagreement |",
                  "|---|---|---|---|---|---|---|"]
    for i, tag in picked:
        query_state, action = states0[i], dict(p_start=p_starts[i], p_stop=p_stops[i])
        fname = f"q{i:04d}_{tag}.png"
        out_path = out_dir / fname
        res = plot_retrieval_debug(query_state, action, predictor, bank, true_next=states1[i],
                                   out_path=out_path, query_id=int(i), start_kind=str(kinds[i]))
        print(f"wrote {out_path}  tag={tag}  moved_mm={res['moved_mm']:.2f}  "
             f"top1_dist={res['top1_dist']:.4f}  knn_disagreement={res['knn_disagreement']:.4f}")
        index_lines.append(f"| `{fname}` | {i} | {kinds[i]} | {tag} | {res['moved_mm']:.2f} | "
                           f"{res['top1_dist']:.4f} | {res['knn_disagreement']:.4f} |")
    (out_dir / "README.md").write_text("\n".join(index_lines) + "\n")
    print("wrote", out_dir / "README.md")


if __name__ == "__main__":
    main()
