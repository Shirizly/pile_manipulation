"""Zoom-window switched-linear foresight (LF-zoom): one linear operator per push-length bin acting on a PUSH-ALIGNED WINDOW of the 300x300 training-convention canvas, at the world-128 pitch (1 mm / px):
window side (mm) = pixel count, so a window pasted back lands exactly on the 128-px world raster. Small sweeps get small windows (64 px -> 16.8M parameters), the largest bin 114 px (169M).
bins: 6 equal-width over [0, 70] mm; window side/res per bin = BIN_RES (>= bin max sweep + 44 mm, as the zoom NFDs). Fit: ridge toward identity per bin on window pairs (w0 -> w1).
LFZoom.rollout(S0,P0s,P1s,canvas0) -> list of pasted 64x64 predictions (the EXP-0074 harness frame), own prediction fed back through the canvas (exactly the zoom-NFD rollout, net replaced by A_b).
LFZoom.rollout128 -> list of cumulative 128-px world-frame predictions (cell-centred 1 mm raster = raster_res(.,128) convention)."""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, "."); sys.path.insert(0, "experiments/EXP-0074-wide-domain-zoom-nfd/code")
from model.zoom_nfd.window_var import extract_windows_v, paste_canvas_v, delta_to_world64_v, _win_idx
from model.zoom_nfd.window_gpu import LO, SIDE
from model.zoom_nfd.rollout import canvas_from_particles
from simple_mpc.adapters import occ_from_particles
dev = "cuda"; BIN_RES = [64, 68, 80, 92, 104, 114]; EDGES = torch.linspace(0, 0.070, 7)


def delta_to_world_cc(delta, P0, P1, side, grid=128, sub=1):
    """window change -> (B,grid,grid) on the cell-centred world grid (pixel i at LO + (i+.5)*SIDE/grid) -- the raster_res / vanilla128 convention."""
    res = delta.shape[-1]; pitch = SIDE / grid; ax = LO + (torch.arange(grid, device=delta.device) + 0.5) * pitch
    pts = torch.stack(torch.meshgrid(ax, ax, indexing="ij"), -1)[None].expand(len(delta), -1, -1, -1)
    return F.grid_sample(delta[:, None], _win_idx(pts, P0, P1, side, res), mode="bilinear", padding_mode="zeros", align_corners=True).squeeze(1)


class LFZoom:
    def __init__(self, ck, device=dev, thr=0.2):
        c = torch.load(ck, map_location="cpu", weights_only=False) if isinstance(ck, str) else ck
        self.ops = [o.to(device).float() for o in c["operators"]]; self.res = c["bin_res"]; self.edges = c["bin_edges"].to(device); self.thr = thr; self.dev = device

    def _bins(self, L):
        return torch.bucketize(L, self.edges[1:-1])

    def _step(self, canvas, P0, P1):
        """-> per-bin lists of (idx, d (n,res,res), side)"""
        L = (P1 - P0).norm(dim=-1); b = self._bins(L); out = []
        for k, A in enumerate(self.ops):
            ix = torch.nonzero(b == k)[:, 0]
            if len(ix) == 0: continue
            res = self.res[k]; side = torch.full((len(ix),), res / 1000.0, device=P0.device); w = extract_windows_v(canvas[ix], P0[ix], P1[ix], side, res)
            y = (A @ w.reshape(len(ix), -1).T).T.reshape(len(ix), res, res); out.append((ix, (y - w), side))
        return out

    @torch.no_grad()
    def rollout(self, S0, P0s, P1s, canvas0=None):
        canvas = (canvas0 if canvas0 is not None else canvas_from_particles(S0, [[0.005] * 3] * S0.shape[1], 300, self.thr)).float().to(self.dev); cur = occ_from_particles(S0).to(self.dev); out = []
        for k in range(P0s.shape[1]):
            P0, P1 = P0s[:, k].to(self.dev), P1s[:, k].to(self.dev); cd = torch.zeros_like(canvas); wd = torch.zeros_like(cur)
            for ix, d, side in self._step(canvas, P0, P1):
                cd[ix] = paste_canvas_v(d, P0[ix], P1[ix], side, canvas.shape[-1]); wd[ix] = delta_to_world64_v(d, P0[ix], P1[ix], side)
            canvas = (canvas + cd).clamp(0, 1); cur = (cur + wd).clamp(0, 1); out.append(cur.cpu())
        return out

    @torch.no_grad()
    def rollout128(self, S0, P0s, P1s, canvas0=None):
        from model.zoom_nfd.world_res import raster_res
        canvas = (canvas0 if canvas0 is not None else canvas_from_particles(S0, [[0.005] * 3] * S0.shape[1], 300, self.thr)).float().to(self.dev); cur = raster_res(S0, [[0.005] * 3] * S0.shape[1], 128).to(self.dev); out = []
        for k in range(P0s.shape[1]):
            P0, P1 = P0s[:, k].to(self.dev), P1s[:, k].to(self.dev); cd = torch.zeros_like(canvas); wd = torch.zeros_like(cur)
            for ix, d, side in self._step(canvas, P0, P1):
                cd[ix] = paste_canvas_v(d, P0[ix], P1[ix], side, canvas.shape[-1]); wd[ix] = delta_to_world_cc(d, P0[ix], P1[ix], side)
            canvas = (canvas + cd).clamp(0, 1); cur = (cur + wd).clamp(0, 1); out.append(cur.cpu())
        return out
