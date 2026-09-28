"""EXP-0047: inventory of existing push transitions for the narrow domain
(n=20 cubes, single-layer start state, 20 mm push, perpendicular blade, chains).

CPU only. One corpus at a time; results/inventory.json rewritten atomically
after every corpus (resume: corpora already in the json are skipped).

Definitions (all measured from tensors, never from config claims):
  push length L   = ||p_stop - p_start||_xy
  exact 20 mm     = |L - 20 mm| < 0.1 mm
  band 18-22 mm   = 18 mm <= L <= 22 mm
  perp deviation  = wrap_pi(angle - (atan2(dy,dx) + pi/2)) in [-90, 90) deg
                    (transforms.functional.action_to_pose convention; blade
                    long axis = plate local x, so its normal is yaw + pi/2)
  single layer    = max particle centre z < FLOOR + 1.5*SIZE - SIZE/2 = 15.0 mm
                    (top face below 1.5 cube sizes above floor; layer-0 centres
                    sit at 12.5 mm, layer-1 at 17.5 mm; floor z = 10 mm)
  clump fraction  = fraction of cubes with an xy neighbour within 1.2*SIZE (6 mm)
  "clump state"   = single layer and clump fraction >= 0.5
  chain link      = row r followed by row r' whose start state equals r's end
                    state (max particle xy diff < 1 mm): same env, next step.
"""
import glob, json, math, os, sys, time, re
from collections import Counter
import torch, yaml

torch.set_num_threads(4)
ROOT = "/home/alon/Code/pile_manipulation"
OUT = f"{ROOT}/experiments/EXP-0047-narrow-domain-inventory/results/inventory.json"
SIZE, FLOOR = 0.005, 0.010
Z_SL = FLOOR + 1.5 * SIZE - SIZE / 2          # 0.015
CLUMP_R = 1.2 * SIZE
LINK_TOL = 1e-3

try:
    Loader = yaml.CUnsafeLoader
except AttributeError:
    Loader = yaml.UnsafeLoader


def row_feats(d):
    s, s_ = d["states"].float(), d["states_"].float()
    ps, pe, a = d["p_starts"].float(), d["p_stops"].float(), d["angles"].float().reshape(-1)
    n = s.shape[1]
    dx, dy = pe[:, 0] - ps[:, 0], pe[:, 1] - ps[:, 1]
    L = torch.hypot(dx, dy)
    phi = torch.atan2(dy, dx)
    dev = torch.remainder(a - (phi + math.pi / 2) + math.pi / 2, math.pi) - math.pi / 2
    dev_deg = dev.abs() * 180 / math.pi
    zmax = s[..., 2].max(1).values
    zmax_ = s_[..., 2].max(1).values
    xy = s[..., :2]
    D = torch.cdist(xy, xy) + torch.eye(n) * 1e3
    clump = (D.min(-1).values < CLUMP_R).float().mean(1)
    return dict(n=n, L=L, dev=dev_deg, sl=zmax < Z_SL, sl_=zmax_ < Z_SL, clump=clump,
                zmax=zmax)


def summarise(F, links, n_rows_total, extra=None):
    """F: dict of concatenated per-row tensors (n20 rows only). links: list of (i,j) indices into F."""
    out = dict(total_rows=n_rows_total)
    if F is None or len(F["L"]) == 0:
        out["n20_rows"] = 0
        return {**out, **(extra or {})}
    L, dev, sl, sl_, cl = F["L"] * 1000, F["dev"], F["sl"], F["sl_"], F["clump"]
    exact = (L - 20).abs() < 0.1
    band = (L >= 18) & (L <= 22)
    p01, p2 = dev < 0.1, dev < 2.0
    slc = sl & (cl >= 0.5)
    out.update(
        n20_rows=int(len(L)),
        n20_single_layer_rows=int(sl.sum()), n20_multilayer_rows=int((~sl).sum()),
        n20_sl_start_and_end=int((sl & sl_).sum()),
        n20_sl_clump_rows=int(slc.sum()), n20_sl_scatter_rows=int((sl & (cl < 0.5)).sum()),
        clump_frac_sl_quantiles=[round(float(q), 3) for q in torch.quantile(cl[sl], torch.tensor([.1, .5, .9]))] if sl.any() else None,
        clump_frac_ml_quantiles=[round(float(q), 3) for q in torch.quantile(cl[~sl], torch.tensor([.1, .5, .9]))] if (~sl).any() else None,
        length_mm_min_mean_max=[round(float(L.min()), 2), round(float(L.mean()), 2), round(float(L.max()), 2)],
        perp_frac_0p1deg=round(float(p01.float().mean()), 4), perp_frac_2deg=round(float(p2.float().mean()), 4),
        rows_exact20=int(exact.sum()), rows_18_22=int(band.sum()),
        sl_exact20=int((sl & exact).sum()), sl_18_22=int((sl & band).sum()),
        sl_exact20_perp0p1=int((sl & exact & p01).sum()), sl_exact20_perp2=int((sl & exact & p2).sum()),
        sl_18_22_perp2=int((sl & band & p2).sum()),
        sl_clump_exact20_perp2=int((slc & exact & p2).sum()), sl_clump_18_22_perp2=int((slc & band & p2).sum()),
    )
    ok_e = sl & exact & p2
    ok_b = sl & band & p2
    if links:
        I = torch.tensor([i for i, _ in links]); J = torch.tensor([j for _, j in links])
        out.update(chain_links=len(links),
                   chain_links_narrow_exact20=int((ok_e[I] & ok_e[J]).sum()),
                   chain_links_narrow_18_22=int((ok_b[I] & ok_b[J]).sum()))
        # longest run of consecutive narrow (18-22) links
    else:
        out.update(chain_links=0, chain_links_narrow_exact20=0, chain_links_narrow_18_22=0)
    return {**out, **(extra or {})}


class Acc:
    def __init__(self):
        self.parts, self.links, self.off, self.total = [], [], 0, 0
        self.ncount = Counter()

    def add(self, d, local_links=()):
        n = d["states"].shape[1]; R = d["states"].shape[0]
        self.total += R; self.ncount[n] += R
        if n != 20:
            return None
        f = row_feats(d)
        self.parts.append({k: v for k, v in f.items() if k != "n"})
        base = self.off
        self.links += [(base + i, base + j) for i, j in local_links]
        self.off += R
        return base

    def cat(self):
        if not self.parts:
            return None
        return {k: torch.cat([p[k] for p in self.parts]) for k in self.parts[0]}


def link_ok(d_a, ra, d_b, rb):
    return (d_a["states_"][ra, :, :2] - d_b["states"][rb, :, :2]).abs().amax(dim=(-1, -2)) < LINK_TOL


def cfg_physics(cfg):
    g = lambda *ks: _get(cfg, ks)
    return (g("material", "friction"), g("box", "friction"), g("material", "density"),
            g("simulation", "settle_steps"), g("safety_margin"),
            g("data_collection", "spawn_mode") or g("spawn", "mode") or g("data_collection", "spawn", "mode"),
            g("data_collection", "pile_aware"), g("data_collection", "push_length"))


def _get(d, ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d if isinstance(d, (int, float, str, bool)) or d is None else str(d)


PHYS_KEYS = ["particle_friction", "box_friction", "density", "settle_steps", "safety_margin",
             "spawn_mode", "pile_aware", "push_length_cfg"]


def scan_files(files):
    """Independent-transition files: chain links at stride n_envs inside each file."""
    acc = Acc(); phys = Counter(); link_diffs = []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        cf = f.replace("_data.pt", "_config.yaml")
        E = None
        if os.path.exists(cf):
            cfg = yaml.load(open(cf), Loader=Loader)
            phys[cfg_physics(cfg)] += d["states"].shape[0]
            E = _get(cfg, ("data_collection", "n_envs")) or _get(cfg, ("statistics", "n_envs"))
        R = d["states"].shape[0]
        links = []
        if E and R > E and d["states"].shape[1] == 20:
            ra = torch.arange(R - E); ok = link_ok(d, ra, d, ra + E)
            link_diffs.append(float(ok.float().mean()))
            links = [(int(i), int(i) + E) for i in ra[ok]]
        acc.add(d, links)
    return acc, phys, link_diffs


def scan_manifest_slates(dirp):
    """same_state_slate_collection: _{k}_data.pt per (slate, step), rows = chains."""
    m = json.load(open(f"{dirp}/manifest.json"))
    by = {(b.get("slate_idx", i), b.get("step_idx", 0)): b["batch_idx"] for i, b in enumerate(m["batches"])}
    acc = Acc(); phys = Counter(); cache = {}
    offs = {}
    for (sl, st), k in sorted(by.items()):
        d = torch.load(f"{dirp}/_{k}_data.pt", map_location="cpu", weights_only=False)
        cf = f"{dirp}/_{k}_config.yaml"
        if os.path.exists(cf):
            phys[cfg_physics(yaml.load(open(cf), Loader=Loader))] += d["states"].shape[0]
        offs[(sl, st)] = acc.add(d)
        cache[(sl, st)] = {"states": d["states"][..., :2].clone(), "states_": d["states_"][..., :2].clone()}
        prev = cache.get((sl, st - 1))
        if prev is not None and offs[(sl, st)] is not None:
            R = d["states"].shape[0]; r = torch.arange(R)
            ok = (prev["states_"] - cache[(sl, st)]["states"]).abs().amax(dim=(-1, -2)) < LINK_TOL
            acc.links += [(offs[(sl, st - 1)] + int(i), offs[(sl, st)] + int(i)) for i in r[ok]]
            del cache[(sl, st - 1)]
    meta = {k: m.get(k) for k in ("n_cubes", "friction", "box_friction", "density", "settle_steps", "spawn_mode",
                                    "push_length", "n_steps", "pile_aware", "placement_aware", "n_envs")}
    return acc, phys, meta


def scan_binned(dirp):
    m = json.load(open(f"{dirp}/manifest.json"))
    acc = Acc(); prev = None; prev_off = None
    ks = sorted(glob.glob(f"{dirp}/step*.pt"))
    for p in ks:
        d = torch.load(p, map_location="cpu", weights_only=False)
        if "simulated" in d:  # PARTIAL corpus
            keep = d["simulated"].bool()
            d = {k: (v[keep] if torch.is_tensor(v) and v.shape[:1] == keep.shape else v) for k, v in d.items()}
        off = acc.add(d)
        if prev is not None:
            R = d["states"].shape[0]; ok = (prev - d["states"][..., :2]).abs().amax(dim=(-1, -2)) < LINK_TOL
            acc.links += [(prev_off + int(i), off + int(i)) for i in torch.arange(R)[ok]]
        prev, prev_off = d["states_"][..., :2].clone(), off
    meta = {k: m.get(k) for k in ("n_cubes", "friction", "box_friction", "density", "settle_steps", "safety_margin",
                                    "spawn_mode", "n_steps", "pile_aware", "complete")}
    meta["step_files"] = [os.path.basename(k) for k in ks]
    return acc, Counter(), meta


def phys_table(phys):
    return [dict(zip(PHYS_KEYS, k), rows=v) for k, v in phys.most_common()]


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    D = f"{ROOT}/Genesis/data"
    jobs = []
    # independent-transition style (recursive globs, *_data.pt only)
    for name, pat in [
        ("overnight_randlen", f"{D}/overnight_randlen/**/*_data.pt"),
        ("Sean_n20", f"{D}/Sean/*/*/cube/n20/**/*_data.pt"),
        ("cube_spectrum_n20", f"{D}/cube_spectrum/n20/*_data.pt"),
        ("granularity_n20", f"{D}/granularity/*20/**/*_data.pt"),
        ("corl_cube_n20_size0.005", f"{D}/corl/cube/n20/size0.005/*_data.pt"),
        ("corl_limited_size0.005", f"{D}/corl_limited/cubes/size0.005/*_data.pt"),
        ("dinowm_test_n20", f"{D}/dinowm_test/cube/n20/**/*_data.pt"),
    ]:
        jobs.append((name, "files", pat))
    for sub in ["n20_heap_5mm"]:
        jobs.append((f"slates/{sub}", "manifest", f"{D}/slates/{sub}"))
    for sub in ["n20_L10mm", "n20_L20mm", "n20_L40mm"]:
        jobs.append((f"slates_multistep/{sub}", "manifest", f"{D}/slates_multistep/{sub}"))
    for p in sorted(glob.glob(f"{D}/slates_binned/*/")):
        jobs.append((f"slates_binned/{os.path.basename(p.rstrip('/'))}", "binned", p.rstrip("/")))
    jobs.append(("DS-0007", "ds7", f"{ROOT}/datasets/DS-0007-sean-same-state-pools/data"))

    for name, kind, arg in jobs:
        if name in res:
            print("skip", name); continue
        t0 = time.time(); print("scan", name, flush=True)
        extra = {}
        if kind == "files":
            files = sorted(f for f in glob.glob(arg, recursive=True) if not f.endswith("_failed.pt"))
            if not files:
                print("  no files"); continue
            acc, phys, ld = scan_files(files)
            extra = dict(path=arg, files=len(files), physics=phys_table(phys),
                         link_frac_per_file_mean=round(sum(ld) / len(ld), 4) if ld else None)
        elif kind == "manifest":
            acc, phys, meta = scan_manifest_slates(arg)
            extra = dict(path=arg, manifest=meta, physics=phys_table(phys))
        elif kind == "binned":
            acc, phys, meta = scan_binned(arg)
            extra = dict(path=arg, manifest=meta)
        elif kind == "ds7":
            acc = Acc(); shards = {}
            for p in sorted(glob.glob(f"{arg}/*_n20.pt")):
                d = torch.load(p, map_location="cpu", weights_only=False)
                acc.add(d); shards[os.path.basename(p)] = int(d["states"].shape[0])
            extra = dict(path=arg, shards_n20=shards, note="derived subset of Sean (same rows); not additional data")
        extra["particle_counts_rows"] = {str(k): v for k, v in acc.ncount.items()}
        res[name] = summarise(acc.cat(), acc.links, acc.total, extra)
        res[name]["scan_s"] = round(time.time() - t0, 1)
        tmp = OUT + ".tmp"; json.dump(res, open(tmp, "w"), indent=1, default=str); os.replace(tmp, OUT)
        print(" ", json.dumps({k: v for k, v in res[name].items() if k not in ("physics", "manifest")}, default=str)[:900], flush=True)


if __name__ == "__main__":
    main()
