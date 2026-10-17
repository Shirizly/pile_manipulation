"""EXP-0076 task space (Genesis-free): scenarios = (object count n, init mode, start state, goal), the 60 % area rule,
the success metric and the physical push time.

Conventions (repo-wide): 64x64 masks indexed mask[x_px, y_px] (row = world x, col = world y), pixel centres at
lo + i (hi - lo) / (G - 1) over the 128 mm tray (OCC_BOUNDS); goal area = mask pixels x pitch^2 (pitch 128/63 mm) --
the same raster the score is computed on. Cube top area = 5 x 5 mm.
Success (user, 2026-10-16): in-goal MASS fraction of the soft-splat scoring raster (simple_mpc.adapters.occ_for_scoring)
>= 0.95. Physical time: EXP-0070 action_time (lift + transit + descent + sweep from the previous plate pose; HOME first).
Start states: chain-start rows of the EXP-0074 Sean TEST-split files (the Sean-trained models never saw them).
"""
import json, os, sys, zlib
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[3]
for p in (REPO, REPO / "experiments/EXP-0074-wide-domain-zoom-nfd/code", REPO / "experiments/EXP-0070-time-optimal-mpc/code"):
    sys.path.insert(0, str(p))
from simple_mpc.adapters import OCC_BOUNDS, OCC_GRID, occ_for_scoring
from Baselines.common.goals import named_goal_mask, mass_in_region

G = OCC_GRID
LO, HI = OCC_BOUNDS["x_min"], OCC_BOUNDS["x_max"]
PITCH = (HI - LO) / (G - 1)                 # 2.032 mm
CUBE_AREA = 0.005 ** 2
RHO_MAX = 0.6                               # user rule: cube area / goal area <= 0.6
GOAL_BOX = 0.054                            # goal pixels must lie within |x|, |y| <= 54 mm (tray inner half-width 64 mm)
RHO_LO = {20: 0.15, 50: 0.30}               # lower end of the sampled area ratio (n50 below 0.3 does not fit the box)
INITS = ("scattered", "inbetween", "piled")
COUNTS = (20, 50)
LETTERS = ("O", "T", "S", "X")
WIDTHS = (20, 15)
SUCCESS = 0.95
CENTRES = (LO + np.arange(G) * PITCH)       # pixel-centre world coordinate along either axis


def mask_area(mask):
    return float(np.asarray(mask).sum()) * PITCH ** 2


def rho_of(n, mask):
    return n * CUBE_AREA / mask_area(mask)


def disc_mask(cx, cy, r):
    X, Y = np.meshgrid(CENTRES, CENTRES, indexing="ij")
    return ((X - cx) ** 2 + (Y - cy) ** 2 <= r * r).astype(np.uint8)


def square_mask(cx, cy, side, theta):
    X, Y = np.meshgrid(CENTRES, CENTRES, indexing="ij"); c, s = np.cos(theta), np.sin(theta)
    u = (X - cx) * c + (Y - cy) * s; v = -(X - cx) * s + (Y - cy) * c
    return ((np.abs(u) <= side / 2) & (np.abs(v) <= side / 2)).astype(np.uint8)


def letter_mask(letter, width, dx_px, dy_px):
    m = named_goal_mask(f"letter_{letter}_w{width}", G, G).astype(np.uint8)
    out = np.zeros_like(m); xs, ys = np.nonzero(m); out[xs + dx_px, ys + dy_px] = 1
    return out


def goal_mask(spec):
    k = spec["kind"]
    if k == "disc":
        return disc_mask(spec["cx"], spec["cy"], spec["r"])
    if k == "square":
        return square_mask(spec["cx"], spec["cy"], spec["side"], spec["theta"])
    if k == "letter":
        return letter_mask(spec["letter"], spec["width"], spec["dx_px"], spec["dy_px"])
    raise KeyError(k)


def in_box(mask):
    xs, ys = np.nonzero(mask)
    return len(xs) > 0 and max(np.abs(CENTRES[xs]).max(), np.abs(CENTRES[ys]).max()) <= GOAL_BOX + 1e-9


def sample_goal(n, kind, rng):
    """Random goal of `kind` that passes the 60 % rule for n cubes and lies inside the goal box. Returns (spec, mask)."""
    for _ in range(1000):
        if kind in ("disc", "square"):
            rho = rng.uniform(RHO_LO[n], RHO_MAX); area = n * CUBE_AREA / rho
            if kind == "disc":
                r = np.sqrt(area / np.pi); lim = GOAL_BOX - r
                if lim <= 0: continue
                spec = dict(kind="disc", r=float(r), cx=float(rng.uniform(-lim, lim)), cy=float(rng.uniform(-lim, lim)))
            else:
                side = np.sqrt(area); th = float(rng.uniform(0, np.pi / 2)); half = side / 2 * (abs(np.cos(th)) + abs(np.sin(th))); lim = GOAL_BOX - half
                if lim <= 0: continue
                spec = dict(kind="square", side=float(side), theta=th, cx=float(rng.uniform(-lim, lim)), cy=float(rng.uniform(-lim, lim)))
        else:
            spec = dict(kind="letter", letter=str(rng.choice(LETTERS)), width=int(rng.choice(WIDTHS)), dx_px=0, dy_px=0)
            m0 = goal_mask(spec); xs, ys = np.nonzero(m0); ok = np.abs(CENTRES) <= GOAL_BOX + 1e-9
            okx = [d for d in range(-G, G) if 0 <= xs.min() + d and xs.max() + d < G and ok[xs.min() + d] and ok[xs.max() + d]]
            oky = [d for d in range(-G, G) if 0 <= ys.min() + d and ys.max() + d < G and ok[ys.min() + d] and ok[ys.max() + d]]
            if not okx or not oky: continue
            spec["dx_px"], spec["dy_px"] = int(rng.choice(okx)), int(rng.choice(oky))
        m = goal_mask(spec)
        if in_box(m) and rho_of(n, m) <= RHO_MAX:
            spec["rho"] = float(rho_of(n, m)); spec["area_mm2"] = mask_area(m) * 1e6
            return spec, m
    raise RuntimeError(f"no feasible {kind} goal for n={n}")


# ---------------------------------------------------------------- start states
def start_pool(n, init, max_files=None):
    """Chain-start states (k, n, 7) of the Sean TEST-split files of shard <init>_n<n>, deterministic order, deduplicated."""
    import sean_data as sd
    sp = sd.split_files(); files = sorted(f for f, s in sp.items() if s == "test" and sd.shard_of(f) == f"{init}_n{n}")[:max_files]
    out, seen = [], set()
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False); S = d["states"].float(); S_ = d["states_"].float()
        A, B = S_[:, :, :2].reshape(len(S), -1), S[:, :, :2].reshape(len(S), -1); D = torch.cdist(A, B); D[torch.arange(len(S)), torch.arange(len(S))] = 9
        is_succ = (D.min(0).values < 1e-4)                       # this row's state is some other row's successor
        for i in (~is_succ).nonzero().flatten().tolist():
            key = tuple(np.round(S[i, :, :2].numpy().ravel(), 4))
            if key in seen: continue
            if S[i, :, :2].abs().max() > HI - 0.0023: continue          # cube centre outside the tray interior (wall contact = 61.5 mm)
            seen.add(key); out.append(dict(file=os.path.relpath(f, REPO), row=i, state=S[i]))
    return out


def make_scenarios(per_stratum, seed=0, kinds=("disc", "square", "letter")):
    """per_stratum scenarios for every (n, init, goal kind); scenario id = f'{n}-{init}-{kind}-{j:03d}'. Deterministic in seed."""
    sc = []
    for n in COUNTS:
        for init in INITS:
            pool = start_pool(n, init)
            for kind in kinds:
                rng = np.random.default_rng(zlib.crc32(f"{seed}-{n}-{init}-{kind}".encode()))
                idx = rng.permutation(len(pool))
                for j in range(per_stratum):
                    st = pool[int(idx[j % len(pool)])]; spec, m = sample_goal(n, kind, rng)
                    S = st["state"]; mf0 = float(mass_frac(S[None], m)[0])
                    sc.append(dict(id=f"n{n}-{init}-{kind}-{j:03d}", n=n, init=init, kind=kind, goal=spec, start_file=st["file"], start_row=st["row"],
                                   state0=S.numpy().round(6).tolist(), mass0=mf0))
    return sc


# ---------------------------------------------------------------- metrics
_MASKS = {}


def mask_t(m):
    key = np.asarray(m).tobytes()
    if key not in _MASKS:
        _MASKS[key] = torch.from_numpy(np.asarray(m).astype(np.float32))
    return _MASKS[key]


@torch.no_grad()
def mass_frac(states, mask):
    """states (T, n, >=3) world -> (T,) in-goal mass fraction of the soft scoring raster."""
    S = torch.as_tensor(states).float()[..., :3]; occ = occ_for_scoring(S); m = mask_t(mask).to(occ.device)
    return (mass_in_region(occ, m) / occ.reshape(len(S), -1).sum(1)).cpu().numpy()


def push_times(actions):
    """list of [sx, sy, ex, ey] -> per-push physical seconds (EXP-0070 action_time from HOME, chained)."""
    from action_time import episode_action_times
    return episode_action_times(actions)["total"] if len(actions) else []


def push_time_from(act, p_prev, first):
    """act (B, 4) torch, p_prev (2,) -> (B,) seconds."""
    from action_time import action_time
    return action_time(act.double(), torch.as_tensor(p_prev, dtype=torch.float64), first)


def time_to_success(actions, masses, thr=SUCCESS):
    """(pushes to success, seconds to success) -- (None, None) if never >= thr. masses (T+1,) incl. the start state."""
    hit = np.nonzero(np.asarray(masses) >= thr)[0]
    if not len(hit):
        return None, None
    k = int(hit[0]); return k, float(sum(push_times(actions[:k]))) if k else 0.0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--per-stratum", type=int, default=30); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "experiments/EXP-0076-expert-demos/results/scenarios.json")); a = ap.parse_args()
    sc = make_scenarios(a.per_stratum, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); tmp = a.out + ".tmp"; json.dump(dict(seed=a.seed, per_stratum=a.per_stratum, scenarios=sc), open(tmp, "w")); os.replace(tmp, a.out)
    import collections
    c = collections.Counter((s["n"], s["init"], s["kind"]) for s in sc); print(len(sc), "scenarios"); print(sorted(c.items()))
    print("rho range", min(s["goal"]["rho"] for s in sc), max(s["goal"]["rho"] for s in sc), "already solved at start:", sum(s["mass0"] >= SUCCESS for s in sc),
          "mean mass0", np.mean([s["mass0"] for s in sc]))
