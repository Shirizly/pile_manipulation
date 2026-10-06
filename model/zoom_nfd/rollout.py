"""model/zoom_nfd/rollout.py -- multi-step rollouts for the zoom-window NFD from a VISUAL state.

State between pushes = a whole-tray canvas (B,C,C) (pixel centre i <-> x = LO + (i+0.5)*128/C mm, dim0 = x). Per push:
    window  = extract_windows_b(canvas)        # rotation + zoom of the model's OWN previous output (differentiable)
    pred    = sigmoid(net([window, plates]))   # next-state window
    canvas += paste_canvas(pred - window)      # the predicted CHANGE is sampled back onto the canvas
The change (not the absolute window) is pasted, so a no-change prediction leaves the canvas exactly as it was and
nothing outside the window is touched. Everything is differentiable, so a model can be trained through T unrolled pushes.
"""
import numpy as np, torch
import torch.nn.functional as F
from model.zoom_nfd.window import WindowSpec
from model.zoom_nfd.window_gpu import raster_world, LO, SIDE, _axes


def canvas_from_particles(states: torch.Tensor, sizes, C: int = 300, thr: float = 0.2, aa: int = 4, chunk: int = 128) -> torch.Tensor:
    """(N,n,7) particle states -> (N,C,C) uint8 training-style canvas (anti-aliased coverage > thr; thr 0.2 reproduces the
    ~28% cube-area inflation of the 1 mm/px training rasters, see NOTES.md)."""
    out = []
    for i in range(0, len(states), chunk):
        out.append((raster_world(states[i:i + chunk], sizes, C, aa=aa) > thr).to(torch.uint8))
    return torch.cat(out)


def extract_windows_b(canvas: torch.Tensor, P0: torch.Tensor, P1: torch.Tensor, spec: WindowSpec, ss: int = 3) -> torch.Tensor:
    """canvas (B,C,C) float; P0,P1 (B,2) metres (same device) -> (B,res,res) windows (ss x ss antialiased)."""
    B, C = canvas.shape[0], canvas.shape[-1]; res = spec.res; pxc = SIDE / C
    u, v = _axes(P0, P1)
    off = (torch.arange(ss, device=canvas.device) - (ss - 1) / 2) / ss
    ci = (torch.arange(res, device=canvas.device)[:, None] + 0.5 + off[None, :]).reshape(-1)
    s = ci * spec.px - spec.margin_back; t = ci * spec.px - spec.side / 2
    Pw = P0[:, None, None, :] + s[None, None, :, None] * u[:, None, None, :] + t[None, :, None, None] * v[:, None, None, :]
    idx = (Pw - LO) / pxc - 0.5
    g = torch.stack([idx[..., 1], idx[..., 0]], -1) / (C - 1) * 2 - 1
    out = F.grid_sample(canvas[:, None], g, mode="bilinear", padding_mode="zeros", align_corners=True)
    return F.avg_pool2d(out, ss).squeeze(1)


def paste_canvas(delta: torch.Tensor, P0: torch.Tensor, P1: torch.Tensor, spec: WindowSpec, C: int) -> torch.Tensor:
    """delta (B,res,res) window change -> (B,C,C) change on the canvas grid (zero outside the window)."""
    B = delta.shape[0]; dev = delta.device; pxc = SIDE / C
    ax = LO + (torch.arange(C, device=dev) + 0.5) * pxc
    pts = torch.stack(torch.meshgrid(ax, ax, indexing="ij"), -1)
    u, v = _axes(P0, P1)
    d = pts[None] - P0[:, None, None, :]
    col = ((d * u[:, None, None, :]).sum(-1) + spec.margin_back) / spec.px - 0.5
    row = ((d * v[:, None, None, :]).sum(-1) + spec.side / 2) / spec.px - 0.5
    g = torch.stack([col, row], -1) / (spec.res - 1) * 2 - 1
    return F.grid_sample(delta[:, None], g, mode="bilinear", padding_mode="zeros", align_corners=True).squeeze(1)


def delta_to_world64(delta: torch.Tensor, P0, P1, spec: WindowSpec, grid: int = 64, sub: int = 3) -> torch.Tensor:
    """delta (B,res,res) -> (B,grid,grid) on the 64-px world grid used by the pasted-frame scores (corner-aligned pixels)."""
    dev = delta.device; pitch = SIDE / (grid - 1)
    ax = torch.arange(grid, device=dev) * pitch + LO
    off = (torch.arange(sub, device=dev) - (sub - 1) / 2) * pitch / sub
    X = (ax[:, None] + off[None, :]).reshape(-1)
    pts = torch.stack(torch.meshgrid(X, X, indexing="ij"), -1)
    u, v = _axes(P0, P1)
    d = pts[None] - P0[:, None, None, :]
    col = ((d * u[:, None, None, :]).sum(-1) + spec.margin_back) / spec.px - 0.5
    row = ((d * v[:, None, None, :]).sum(-1) + spec.side / 2) / spec.px - 0.5
    g = torch.stack([col, row], -1) / (spec.res - 1) * 2 - 1
    return F.avg_pool2d(F.grid_sample(delta[:, None], g, mode="bilinear", padding_mode="zeros", align_corners=True), sub).squeeze(1)


def load_chains(pattern: str, require_valid: bool = True):
    """Chain files (chain_env/chain_step, rows chained exactly: states_[e,s] == states[e,s+1]) -> dict of arrays over complete chains:
    S (E,T,n,7) state before push s, S_ after, P0/P1 (E,T,2) metres, kind (E,), valid (E,T). Chains with a missing step are dropped;
    with require_valid also chains with any invalid / non-single-layer step (so every kept chain is fully usable)."""
    import glob
    S, S_, P0, P1, K, V = [], [], [], [], [], []
    for f in sorted(glob.glob(pattern)):
        d = torch.load(f, map_location="cpu", weights_only=False)
        env, step = d["chain_env"].numpy(), d["chain_step"].numpy()
        for e in np.unique(env):
            ix = np.nonzero(env == e)[0]; ix = ix[np.argsort(step[ix])]
            if (step[ix] != np.arange(len(ix))).any(): continue
            ok = d["valid"][ix].bool() & d["single_layer"][ix].bool() if "single_layer" in d else d["valid"][ix].bool()
            if require_valid and not bool(ok.all()): continue
            S.append(d["states"][ix].float()); S_.append(d["states_"][ix].float())
            P0.append(d["p_starts"][ix, :2].float()); P1.append(d["p_stops"][ix, :2].float()); K.append(d["start_kind"][ix[0]]); V.append(ok)
    n = min(len(x) for x in S)
    T = min(len(x) for x in S)
    return dict(S=torch.stack([x[:T] for x in S]), S_=torch.stack([x[:T] for x in S_]), P0=torch.stack([x[:T] for x in P0]),
                P1=torch.stack([x[:T] for x in P1]), kind=np.array(K), valid=torch.stack([x[:T] for x in V]))
