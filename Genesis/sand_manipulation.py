"""
Genesis/sand_manipulation.py — the same scene, with sand instead of cubes.

What changes and what does not
------------------------------
Everything except the manipulated material is deliberately identical to
`SandboxManipulation`: the same tray walls, the same plate geometry and
actuator model, the same trapezoidal sweep, the same `execute_action`, and the
same pile-aware action sampling. This class **subclasses** rather than
reimplements so that stays true by construction — the scene furniture is built
by the parent's own `_add_entities`, with only the granular block swapped out.

The material becomes a Genesis MPM continuum (`gs.materials.MPM.Sand`), which
forces four differences:

1. **The scene needs `mpm_options`.** The parent builds a rigid-only scene, so
   `_init_scene` is overridden to add an MPM solver with a domain that brackets
   the tray.
2. **Particle count is an output, not an input.** MPM samples the initial volume
   at its own `particle_size`, so the count is read back after the entity is
   added rather than requested. `_material_params["n_particles"]` is set from it
   before the parent allocates its state buffers.
3. **There are no orientations.** State is kept in the parent's ``(B, N, 7)``
   layout with identity quaternions rather than a new ``(B, N, 3)`` one, so
   every downstream consumer — the dataset loader, `variance_decomposition.py`
   (which reads `states[..., :3]`), the occupancy conversions — works unchanged.
   Sand grains genuinely have no meaningful orientation, so the identity quat is
   honest padding, not a placeholder for something missing.
4. **Reset is a pose write, not a respawn.** There is no rejection sampling to
   redo; the initial sampled configuration is captured once at build time and
   restored, which is also far cheaper than re-settling.

Plate height
------------
The cube path parks the plate half a particle above the floor, because a 5 mm
cube's centre sits there. Sand has no such standoff: material that the blade
rides over is material it does not push, and with grains ~2 mm across a 2.5 mm
gap would let most of the pile pass underneath. So `_operation_height` is
overridden to put the blade's *bottom* one MPM particle radius above the tray
floor — tight, but not interpenetrating, which the MPM/rigid coupler handles
badly. Set `plate.sweep_clearance` in the config to override.

Occupancy
---------
Do **not** feed sand through `particles_to_occupancy`: it clamps to a binary
silhouette and would discard the depth information that is the entire point of a
continuum. Use `transforms/sand_occupancy.py`.
"""

from __future__ import annotations

import types
from pathlib import Path

import genesis as gs
import torch
import yaml

from .sandbox_manipulation_clean import SandboxManipulation


class _SceneKwargCapture:
    """Stands in for `gs.Scene` just long enough to record the parent's kwargs.

    The parent builds its scene in a single `gs.Scene(...)` call with no hook
    for extra solver options, and reading the options back off a *built* scene
    means guessing at attribute names that Genesis is free to change. Capturing
    the call instead means the sand scene inherits every rigid/sim/vis option
    the parent chooses, including ones added later, with nothing duplicated.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.profiling_options = types.SimpleNamespace(show_FPS=True)


class SandManipulation(SandboxManipulation):
    """`SandboxManipulation` with an MPM sand continuum as the material."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The parent computes _operation_height inline from the rigid particle
        # size, so it cannot be overridden through a method; correct it here,
        # along with the servo target that was derived from it.
        self._operation_height = self._compute_operation_height()
        self._horizontal_dof_fix[:, 0] = self._operation_height
        self._log(f"sand: blade bottom {1000 * (self._operation_height - self._wall_thickness / 2 - self._plate_params['size'][2] / 2):.2f} mm "
                  f"above the tray floor")

    # ------------------------------------------------------------------ #
    # Scene construction
    # ------------------------------------------------------------------ #

    def _init_scene(self):
        """Rigid scene exactly as the parent builds it, plus an MPM solver."""
        mpm_cfg = self._config.get("mpm_options", {}) or {}

        # MPM is far more substep-hungry than the rigid solver: Genesis warns
        # when substep_dt exceeds a CFL-style bound derived from the grid
        # spacing and the material's wave speed, and past it the sim goes
        # unstable rather than merely inaccurate. The rigid path's 5 substeps
        # leave it ~10x over. Raising substeps rather than lowering dt is
        # deliberate: dt sets the plate's control cadence, so lowering it would
        # change the sweep trajectory and the scene would no longer be identical
        # to the cube setup.
        self._config["simulation"]["substeps"] = int(
            self._config["simulation"].get("substeps_mpm", 30))
        self._mpm_particle_size = float(mpm_cfg.get("particle_size", 0.002))

        real_scene = gs.Scene
        gs.Scene = _SceneKwargCapture
        try:
            super()._init_scene()
        finally:
            gs.Scene = real_scene
        kwargs = dict(self._scene.kwargs)

        width, depth, height = self._box_params["vol"]

        # grid_density is cells per METRE, not cells per domain. Genesis' own
        # default of 64 means dx = 15.6 mm, which is coarser than the whole pile
        # and eight times the grain size -- MPM would resolve nothing. The
        # standard MPM ratio is ~2 particles per cell per axis, so dx is derived
        # from the grain size unless the config overrides it.
        dx_target = 2.0 * self._mpm_particle_size
        grid_density = int(mpm_cfg.get("grid_density", 0)) or int(round(1.0 / dx_target))
        dx = 1.0 / grid_density

        # THE MPM DOMAIN IS THE CONTAINER. Measured: Genesis' fixed rigid boxes
        # do NOT couple to MPM under either the default or the Legacy coupler --
        # sand falls straight through the tray floor and settles on the domain
        # boundary at z = lower + 3*dx. (A *moving* rigid body does couple, which
        # is why the plate still pushes sand; the exclusion appears to be
        # specific to fixed geoms.) So rather than fight it, the domain is sized
        # so that its own boundary, after the solver's 3*dx padding, coincides
        # exactly with the tray's inner walls and floor. The sand is then
        # contained by the boundary condition instead of by the wall geometry,
        # at the same physical location.
        #
        # The rigid tray entities are still built, unchanged, by the parent --
        # they are what the plate and any rigid debris collide with, and keeping
        # them means the scene really is identical apart from the material.
        bpad = 3.0 * dx
        wall = float(self._wall_thickness)
        lower = tuple(mpm_cfg.get("lower_bound",
                                  (-width / 2 - bpad, -depth / 2 - bpad,
                                   wall / 2 - bpad)))
        upper = tuple(mpm_cfg.get("upper_bound",
                                  (width / 2 + bpad, depth / 2 + bpad,
                                   height + bpad)))

        cells = [int(round((u - l) / dx)) for l, u in zip(lower, upper)]
        self._log(f"sand: MPM grid dx {1000 * dx:.1f} mm "
                  f"({cells[0]}x{cells[1]}x{cells[2]} = {cells[0]*cells[1]*cells[2]/1e6:.2f}M cells); "
                  f"domain boundary coincides with the tray "
                  f"(floor z = {1000 * wall / 2:.0f} mm); "
                  f"CPIC {'on' if bool(mpm_cfg.get('enable_CPIC', True)) else 'OFF'}")

        # Without an explicitly rigid-MPM-capable coupler the sand does not see
        # the tray at all: it falls straight through the floor and settles on the
        # MPM domain boundary. That failure is quiet -- the run completes, the
        # data looks like data -- and it was caught only by noticing grains at
        # z = -8 mm with the tray floor at +10 mm. LegacyCoupler is the one that
        # implements rigid_mpm.
        kwargs["coupler_options"] = gs.options.LegacyCouplerOptions()

        # CPIC (Compatible Particle-In-Cell) is what makes a THIN rigid body
        # behave like a barrier to a continuum. Without it the grid transfer is
        # blind to the blade: particles on opposite sides of it share grid
        # nodes, so material behind the plate is dragged along by material in
        # front, which looks exactly like sand having adhesion. The sand model
        # itself has none -- Genesis implements the Klar Drucker-Prager return
        # mapping, which releases all elastic deformation when the trace turns
        # tensile, so the constitutive tensile strength is exactly zero. Any
        # pulling that shows up is therefore numerical, and this is its cure.
        # Off by default in Genesis; unsupported only in differentiable mode,
        # which this path does not use.
        enable_cpic = bool(mpm_cfg.get("enable_CPIC", True))
        kwargs["mpm_options"] = gs.options.MPMOptions(
            particle_size=self._mpm_particle_size,
            grid_density=grid_density,
            lower_bound=lower,
            upper_bound=upper,
            enable_CPIC=enable_cpic,
        )
        self._scene = gs.Scene(**kwargs)
        self._scene.profiling_options.show_FPS = False

    def _add_entities(self):
        """Parent's tray and plate verbatim; sand in place of the cubes.

        `random_sequential_addition` is stubbed for the duration of the parent
        call rather than the box/plate construction being copied here. Copying
        would be the obvious move and it is the wrong one: the two scenes would
        drift the moment either is tuned, and "the rest of the scene is
        identical" is the whole premise of this comparison.
        """
        import Genesis.sandbox_manipulation_clean as _parent_mod

        real_rsa = _parent_mod.random_sequential_addition
        _parent_mod.random_sequential_addition = (
            lambda **_kw: ([], []))          # no rigid grains
        try:
            super()._add_entities()
        finally:
            _parent_mod.random_sequential_addition = real_rsa

        self.sand = self._add_sand()

        # MPM decides the count by sampling the volume; the parent's buffers are
        # sized from n_particles, so publish it before they are allocated.
        n = int(self.sand.n_particles)
        self._material_params["n_particles"] = n
        self._n_active = n
        # `self.material` is only ever used for its length and for per-entity
        # property setting, neither of which applies to a single MPM body.
        self.material = [self.sand] * n
        self._log(f"sand: {n} MPM particles at {1000 * self._mpm_particle_size:.1f} mm")

    def _add_sand(self):
        """A single sand pile at the tray centre.

        Cylinder rather than box: a pile is round, and a square column has
        corners that shed material asymmetrically on the first settle, which
        shows up as a direction-dependent bias in the very first transition of
        every episode.
        """
        sand_cfg = self._config.get("sand", {}) or {}
        wall = float(self._wall_thickness)
        radius = float(sand_cfg.get("pile_radius", 0.02))
        height = float(sand_cfg.get("pile_height", 0.012))
        floor_z = wall / 2.0

        material = gs.materials.MPM.Sand(
            E=float(sand_cfg.get("E", 1.0e6)),
            nu=float(sand_cfg.get("nu", 0.2)),
            rho=float(sand_cfg.get("rho", 1500.0)),
            friction_angle=float(sand_cfg.get("friction_angle", 45.0)),
            sampler=str(sand_cfg.get("sampler", "pbs")),
        )
        morph = gs.morphs.Cylinder(
            pos=(0.0, 0.0, floor_z + height / 2.0),
            radius=radius,
            height=height,
        )
        return self._scene.add_entity(material=material, morph=morph)

    # ------------------------------------------------------------------ #
    # Geometry overrides
    # ------------------------------------------------------------------ #

    def _compute_operation_height(self) -> float:
        """Blade bottom one MPM particle radius above the tray floor.

        The cube path leaves half a cube of standoff because that is where a
        cube's centre sits. Sand needs the blade essentially on the floor:
        anything the blade rides over is material it does not push, and at
        ~2 mm grains a cube-sized gap would let most of a pile pass underneath.
        A radius of clearance keeps the blade from being born interpenetrating
        the sand, which the MPM/rigid coupler resolves violently.
        """
        clearance = float(self._plate_params.get(
            "sweep_clearance", self._mpm_particle_size / 2.0))
        return (self._wall_thickness / 2.0
                + clearance
                + self._plate_params["size"][2] / 2.0)

    # ------------------------------------------------------------------ #
    # State: positions only, padded to the parent's (B, N, 7) layout
    # ------------------------------------------------------------------ #

    def _cache_particle_idx(self):
        """No per-grain rigid links or dofs exist for a continuum."""
        self._particle_links_idx = torch.zeros(0, dtype=torch.long, device=gs.device)
        self._particle_dofs_idx = torch.zeros(0, dtype=torch.long, device=gs.device)

    def _pile_motion(self, quantile=None):
        """Grain speed, from the MPM solver rather than rigid dofs.

        The parent reads `rigid_solver.get_dofs_velocity` over per-grain dof
        indices and short-circuits to ``(0.0, 0.0)`` when there are none. For
        sand there are none, so without this override `_pile_is_at_rest` would
        be True on the first check and **the settle loop would exit after a
        single step** -- sand would never settle, silently, and every recorded
        `s'` would be a snapshot of material still in motion.

        Angular velocity is reported as 0: MPM grains carry no orientation, so
        the parent's angular threshold is vacuous here rather than merely
        unused.
        """
        vel = self.sand.get_particles_vel()
        if vel.dim() == 2:
            vel = vel.unsqueeze(0)
        lin = vel.norm(dim=-1).flatten()
        if quantile is None:
            return float(lin.max()), 0.0
        q = torch.tensor(quantile, device=lin.device, dtype=lin.dtype)
        return float(torch.quantile(lin, q)), 0.0

    def _get_particle_positions(self):
        pos = self.sand.get_particles_pos()
        if pos.dim() == 2:                       # (N, 3) when n_envs == 1
            pos = pos.unsqueeze(0)
        return pos

    def _get_particle_quats(self):
        """Identity for every grain — sand has no orientation.

        Kept in the state so the ``(B, N, 7)`` layout, and therefore every
        downstream consumer, is shared with the rigid path.
        """
        n = self._material_params["n_particles"]
        q = torch.zeros((self._n_envs, n, 4), device=gs.device)
        q[..., 0] = 1.0
        return q

    def _write_particle_poses(self, pos, quat, envs_idx=None):
        """Positions only; quaternions are ignored (see `_get_particle_quats`)."""
        self.sand.set_particles_pos(pos)
        self.sand.set_particles_vel(torch.zeros_like(pos))

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #

    _state_library = None

    def build(self):
        super().build()
        self.update_material_state()
        # The as-sampled pile, captured once. Resetting to this is both correct
        # (it is a valid settled configuration) and far cheaper than re-running
        # a settle, which is what dominates cube collection.
        self._initial_sand_pos = self._get_particle_positions().detach().clone()
        self._log("sand: captured initial pile for resets")

    def set_state_library(self, states: torch.Tensor | None) -> None:
        """Draw episode starts from a bank of varied piles instead of one pile.

        Without this every episode restarts from the identical as-sampled pile,
        because MPM samples its volume once. Measured consequence: the canonical
        input states spanned only ~14 dimensions for 90% of their variance,
        which caps the rank any fitted operator can need and makes a low-rank
        result partly an artefact of the data. See
        `Genesis/sand_state_library.py` for how a varied bank is built from
        states already collected.

        ``states`` is ``(K, N, 3)``; N must match this scene's grain count.
        """
        if states is None:
            self._state_library = None
            return
        n = self._material_params["n_particles"]
        if states.shape[1] != n:
            raise ValueError(
                f"state library has {states.shape[1]} grains, scene has {n}. "
                f"A library is specific to the MPM particle_size and pile "
                f"volume it was sampled at.")
        self._state_library = states.to(gs.device)
        self._log(f"sand: episode starts will be drawn from {states.shape[0]} "
                  f"library states")

    def shuffle_particles(self, pile_extent=None, pile_layers=None, spawn_mode=None):
        """Start a new episode: draw from the state library, or restore the
        as-sampled pile if there is none.

        Takes and ignores the cube spawn arguments so callers written against
        the rigid path (including `collect_data_samples`) work unchanged.
        """
        self.flush_transitions()
        self.set_transition_context(None)

        lib = getattr(self, "_state_library", None)
        if lib is not None:
            idx = torch.randint(0, lib.shape[0], (self._n_envs,), device=gs.device)
            self._write_particle_poses(lib[idx], None)
            return
        if not hasattr(self, "_initial_sand_pos"):
            return
        self._write_particle_poses(self._initial_sand_pos, None)

    def set_particle_state(self, pos: torch.Tensor, quat: torch.Tensor) -> None:
        self._write_particle_poses(pos[..., :3], None)

    # ------------------------------------------------------------------ #
    # Material properties
    # ------------------------------------------------------------------ #

    def set_material_properties(self, setting):
        """MPM material parameters are fixed at entity creation.

        Genesis builds the constitutive model into the solver when the entity is
        added, so friction/density cannot be re-set per batch the way a rigid
        body's can. Accepted and ignored so the collection loop is shared; vary
        them by constructing a new scene with different ``sand:`` config, and
        note it in the run's saved config rather than assuming a sweep happened.
        """
        self._config["data_collection"].setdefault("sampled", {})
        self._log("sand: material properties are fixed at build; "
                  "set_material_properties is a no-op")


def load_sand_config(path: str | Path = "configs/sand.yaml") -> dict:
    """Read a sand config from `Genesis/configs/`."""
    full = Path(__file__).parent / path
    with open(full) as fh:
        return yaml.safe_load(fh)
