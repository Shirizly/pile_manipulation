"""
Genesis/spawn_geometry.py — particle spawn layouts that the RSA draw cannot make.

Why this exists
---------------
The default spawn is rejection sampling over an xy region, optionally split into
stacked layers that are then dropped. That reliably produces a *dense monolayer*,
however hard it is pushed: measured, 90% of particles end in layer 0 at 30 cubes
of 5 mm and 94% at 80 cubes of 3 mm, and raising friction from 0.3 to 0.9 made it
slightly flatter rather than deeper. The cause is the drop itself — cubes released
above the floor bounce and spread outward before they come to rest, and lighter
cubes bounce more.

Depth is the untested half of two hypotheses about why the paper's per-pixel
linear operator worked and ours did not (docs/linear_foresight_findings.md §4,
H1 and H2), so it needs a mechanism that does not involve dropping anything.

A stepped pyramid does: every cube rests on the one below, so the structure starts
at rest under gravity and has no bounce energy to spread with. Measured at
50 cubes of 5 mm, mean layer index (mass-weighted, 0 = flat):

    dropped spawn      0.05 - 0.09     footprint 65-67 mm
    pyramid            0.68            footprint 23 mm
    pyramid, settled   0.68            footprint 23 mm   (identical -- nothing moves)
    pyramid + 1 push   0.67 - 0.68     footprint 38-48 mm

So it is ~7x deeper than the dropped spawn by mean layer, a third of the
footprint, does not relax when gravity is applied, and keeps its layering through
a push.

Caveat on those numbers: they were measured with a [25, 16, 9] layout for n=50.
`pyramid_layer_plan` was then changed to fix a non-monotonicity (see its
docstring), and now gives [36, 9, 4, 1] for n=50 -- one layer *deeper*, so it
should be at least as good, but it has not been re-measured. Re-run
`scripts/probe_pile_depth.py --counts 50` to confirm before relying on the exact
figures.

Pure torch, no `genesis` import, so it is unit testable without a GPU
(tests/test_spawn_geometry.py).
"""

from __future__ import annotations

import itertools

import torch


def pyramid_layer_plan(n: int) -> list[int]:
    """How many cubes go in each layer of an n-cube pyramid, bottom first.

    Builds the tallest *complete* pyramid that fits — layers ``k^2, (k-1)^2, ...,
    1`` for the largest ``k`` with ``sum(i^2) <= n`` — then adds whatever is left
    over to the **bottom** layer, widening the base rather than perching an
    unstable partial layer on the apex:

        n=14 -> [9, 4, 1]           three layers
        n=30 -> [16, 9, 4, 1]       four layers
        n=50 -> [36, 9, 4, 1]       four layers  (30 complete + 20 into the base)
        n=55 -> [25, 16, 9, 4, 1]   five layers
        n=80 -> [50, 16, 9, 4, 1]   five layers

    The layer count is then **monotone** in ``n``, which the obvious alternative
    is not: choosing the base from the smallest full-pyramid sum that *reaches*
    ``n`` and truncating the top gives 5 layers at n=55 but only 2 at n=56
    (``[36, 20]``), because the base jumps a whole width to absorb one extra
    cube. A unit test pins the monotonicity.

    Placing only a complete pyramid and parking the remainder was tried first and
    is a trap: parked cubes have to go somewhere, and anywhere outside the tray
    is either below the floor (they are ejected violently) or inside the
    measurement (they register as a spurious bottom layer — observed at 40% of a
    50-cube run, with the reported footprint inflated to 751 mm).
    """
    if n <= 0:
        return []
    k = 1
    while sum(i * i for i in range(1, k + 2)) <= n:
        k += 1
    plan = [i * i for i in range(k, 0, -1)]
    plan[0] += n - sum(plan)
    return plan


def pyramid_positions(n: int, size: float, gap: float = 1.15,
                      centre: tuple[float, float] = (0.0, 0.0),
                      floor_z: float = 0.0,
                      device=None, dtype=torch.float32):
    """Centre positions for an n-cube stepped pyramid resting on ``floor_z``.

    Parameters
    ----------
    n       : cubes to place. All of them are placed (see `pyramid_layer_plan`).
    size    : cube edge length, metres.
    gap     : lateral pitch as a multiple of ``size``. Slightly above 1 so
              neighbours are not born in contact, which the solver dislikes.
    centre  : xy centre of the pyramid.
    floor_z : z of the surface the bottom layer sits on.

    Returns
    -------
    ``(positions, n_layers)`` with positions ``(n, 3)``.

    Layers are square blocks, largest at the bottom, each centred on ``centre``,
    so the whole structure is symmetric. A caller wanting to avoid a perfectly
    symmetric (and therefore unstably balanced) start should add a small xy
    jitter -- `SandboxManipulation.shuffle_particles` adds 8% of a cube width.
    """
    plan = pyramid_layer_plan(n)
    pitch = size * gap
    pos = []
    for i, take in enumerate(plan):
        side = int(round(take ** 0.5))
        if side * side < take:
            side += 1
        z = floor_z + size * (0.5 + i) * 1.001
        off = (side - 1) / 2.0
        cells = list(itertools.product(range(side), repeat=2))[:take]
        for a, b in cells:
            pos.append((centre[0] + (a - off) * pitch,
                        centre[1] + (b - off) * pitch,
                        z))
    return torch.tensor(pos, dtype=dtype, device=device), len(plan)


def stagger_layers(pos: torch.Tensor, size: float, gap: float = 1.15,
                   stagger: float = 0.5, floor_z: float = 0.0) -> torch.Tensor:
    """Offset alternate layers of a pyramid sideways ("brick bond").

    `pyramid_positions` gives every layer the same pitch and the same centre, so
    an upper cube rests squarely on exactly ONE lower cube: the structure is a
    stack of columns. That is extremely stable, and measurably so — probed at
    n=20 and n=50 with 3 mm cubes, the settled layer occupancy came out exactly
    equal to the placed layer plan (75/20/5 and 72/18/8) under every
    perturbation tried, including dropping the whole stack a full cube height.
    It slid sideways as a unit and never restructured, so every episode would
    start from the same lattice.

    Shifting odd layers by half a pitch makes an upper cube bridge the seam
    between four lower ones. It then has somewhere to fall, and the pile
    collapses into an irregular heap while keeping the depth that only placement
    can give (a dropped spawn leaves 90-94% of cubes in layer 0).

    Parameters
    ----------
    pos      : ``(n, 3)`` from `pyramid_positions`.
    stagger  : offset in pitches. 0 leaves ``pos`` untouched; 0.5 is the
               half-pitch brick bond.
    floor_z  : the ``floor_z`` passed to `pyramid_positions`, needed to recover
               each cube's layer index from its z.

    Returns a new tensor; ``pos`` is not modified.
    """
    if stagger == 0.0:
        return pos.clone()
    # Layer index back out of z. pyramid_positions places layer i at
    # floor_z + size * (0.5 + i) * 1.001, so rounding is safe.
    layer = ((pos[:, 2] - floor_z - 0.5 * size) / size).round()
    odd = (layer % 2 == 1).unsqueeze(-1)
    out = pos.clone()
    out[:, :2] = out[:, :2] + odd * (stagger * size * gap)
    return out


def heap_positions(n: int, size: float, gap: float = 1.15,
                   base_frac: float = 0.6,
                   centre: tuple[float, float] = (0.0, 0.0),
                   floor_z: float = 0.0, generator=None,
                   device=None, dtype=torch.float32):
    """An IRREGULAR two-layer heap: random lattice sites, not a stepped pyramid.

    Why this exists rather than just perturbing `pyramid_positions`
    ----------------------------------------------------------------
    A pyramid gives depth but no variety, and no amount of perturbation fixes
    that. Probed at 3 mm across eight setups spanning lateral jitter, cube yaw,
    a full-cube drop and a half-pitch brick bond, the settled layer occupancy
    came out EXACTLY equal to the placed layer plan every time (75/20/5 at
    n=20, 72/18/8 at n=50); the stack slid a millimetre or three as a unit and
    never restructured. That is not a tuning failure but the physics of
    flat-faced rigid cubes: they stack stably in almost any arrangement, and a
    cube bridging two supports is as stable as one sitting on a single support.
    Sand collapses into an irregular heap on its own; a cube pile does not.

    So the irregularity has to be PLACED, not waited for. Layer 0 takes a
    random subset of sites in a square footprint, and each remaining cube goes
    on top of a randomly chosen occupied site. Every draw is a different heap,
    it is two layers deep by construction, and it is stable — so it stays put
    through the settle instead of exploding into a monolayer the way a dropped
    spawn does (90-94% of cubes in layer 0).

    Parameters
    ----------
    base_frac : fraction of the cubes to put in layer 0. 0.6 leaves 40% on top,
        which is a visibly lumpy two-layer heap rather than a paved floor with
        a few strays.
    generator : optional torch.Generator, for reproducible draws.

    Returns ``(positions, n_layers)`` with positions ``(n, 3)``, matching
    `pyramid_positions` so the two are interchangeable at the call site.
    """
    if n <= 0:
        return torch.zeros((0, 3), dtype=dtype, device=device), 0

    n0 = max(1, min(n, int(round(base_frac * n))))
    # Footprint just big enough to hold layer 0 with gaps left over, so the
    # random subset is genuinely holey rather than a solid slab.
    side = 1
    while side * side < n0 * 1.4:
        side += 1

    # EVERY tensor below names device="cpu" explicitly and the result is moved
    # at the end. Genesis sets torch's DEFAULT device to cuda, so a bare
    # torch.arange here silently lands on the GPU and the final cat dies with
    # "Expected all tensors to be on the same device" -- the same trap that hit
    # Genesis/sand_state_library.py.
    def _rand(k, hi):
        return torch.randperm(hi, generator=generator, device="cpu")[:k]

    pitch = size * gap
    off = (side - 1) / 2.0
    base_sites = _rand(n0, side * side)
    ax, ay = base_sites // side, base_sites % side

    xs = [centre[0] + (ax.to(dtype) - off) * pitch]
    ys = [centre[1] + (ay.to(dtype) - off) * pitch]
    zs = [torch.full((n0,), floor_z + size * 0.5 * 1.001, dtype=dtype,
                     device="cpu")]

    n1 = n - n0
    if n1 > 0:
        # Each upper cube sits on an occupied base site. Sampling WITHOUT
        # replacement where possible keeps them from being asked to occupy the
        # same spot; beyond that, wrap round (a third layer is fine, and is
        # what makes tall lumps).
        pick = torch.cat([_rand(min(n0, n1 - i * n0), n0)
                          for i in range((n1 + n0 - 1) // n0)])[:n1]
        lvl = torch.arange(n1, device="cpu") // n0 + 1
        xs.append(centre[0] + (ax[pick].to(dtype) - off) * pitch)
        ys.append(centre[1] + (ay[pick].to(dtype) - off) * pitch)
        zs.append(floor_z + size * (0.5 + lvl.to(dtype)) * 1.001)

    pos = torch.stack([torch.cat(xs), torch.cat(ys), torch.cat(zs)], dim=-1)
    n_layers = 1 + (1 if n1 > 0 else 0) + (1 if n1 > n0 else 0)
    return pos.to(device=device, dtype=dtype), n_layers
