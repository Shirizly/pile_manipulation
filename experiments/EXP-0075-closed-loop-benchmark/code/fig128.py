"""EXP-0075: predictions and truth at the models' NATIVE 128 px (the planner's objective still uses the 64x64 pasted frame; these are for figures / error inspection only).
predict128(model, S0, seq) -> (H+1, 128, 128): [input raster, state predicted after push 1..H] of the WideEns (zoom canvas down to 128 + vanilla128 state, member mean, same per-push mass balance as WideEns._predict).
truth128(parts) -> (n, 128, 128) exact hard raster (raster_res) of simulator particle states (n, 20, 7)."""
import torch, torch.nn.functional as F
import wide_planner as wp
from wide_planner import DEV, fix_delta_balance, SIZES
from model.zoom_nfd.window_var import side_for, plates_v, extract_windows_v, paste_canvas_v
from model.zoom_nfd.world_res import raster_res, plates_res
import eval_wide as ew


def _down(canvas):                                   # (N,300,300) -> (N,128,128), area average
    return F.interpolate(canvas[:, None].float(), size=(128, 128), mode="area").squeeze(1)


@torch.no_grad()
def predict128(model, S0, seq):
    seq = seq.to(DEV).float()[None]; P0, P1 = seq[..., :2], seq[..., 2:]; H = seq.shape[1]; model.begin(S0); per_member = []
    for m in model.members:
        fr = []
        if m.kind == "zoom":
            canvas = m.canvas0.expand(1, -1, -1); fr.append(_down(canvas))
            for k in range(H):
                p0, p1 = P0[:, k], P1[:, k]; side = side_for((p1 - p0).norm(dim=-1)); w = extract_windows_v(canvas, p0, p1, side, m.res)
                d = torch.sigmoid(m.net(torch.cat([w[:, None], plates_v(p0, p1, side, m.res)], 1))).squeeze(1) - w
                canvas = (canvas + paste_canvas_v(d, p0, p1, side, m.C)).clamp(0, 1); fr.append(_down(canvas))
        else:
            st = m.st0.expand(1, -1, -1); fr.append(st)
            for k in range(H):
                st = torch.sigmoid(m.net(torch.cat([st[:, None], plates_res(P0[:, k], P1[:, k], m.res)], 1))).squeeze(1); fr.append(st)
        per_member.append(torch.cat(fr))
    raw = torch.stack(per_member).mean(0)            # (H+1,128,128)
    if not model.balance:
        return raw.cpu()
    out = [raw[0]]; prev = raw[0]
    for k in range(1, H + 1):
        prev = (prev + fix_delta_balance(raw[k] - raw[k - 1])).clamp(0, 1); out.append(prev)
    return torch.stack(out).cpu()


def truth128(parts, inflated=True):
    """inflated=True: training-convention raster (canvas_from_particles thr 0.2 = ~28 % cube-area inflation, down to 128), the frame the models' own outputs live in; False: exact cube footprints (raster_res)."""
    p = parts[:, :, :7].float().cpu()
    if not inflated:
        return raster_res(p, SIZES(p.shape[1]), 128)
    from model.zoom_nfd.rollout import canvas_from_particles
    return _down(canvas_from_particles(p, SIZES(p.shape[1]), 300, 0.2).float()).cpu()
