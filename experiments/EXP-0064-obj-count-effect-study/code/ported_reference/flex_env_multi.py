"""Batched FlexEnv: N identical copies of one task environment in a single solver.

Motivation: a single carrot-pile env leaves the GPU badly underutilized -- the
Flex solver is launching kernels over a few thousand particles per step.  Laying
out N copies of the same env on a grid inside one scene lets one `pyflex.step()`
advance all N at once, so per-action cost grows much slower than N.

What is shared and what is not:
  - shared: one solver, one GL context, one `pyflex.step()`, one camera rig
  - per env: its own 4 container walls, its own Franka (its own pybullet body),
    its own slice of the particle buffer, its own action

The tiles are exact replicas by construction, not by copying: scene 24
(`yx_CarrotsGrid`, PyFleX/bindings/scenes/yx_carrots_grid.h) re-seeds both RNGs
before each tile, so every tile generates the same carrot bodies -- same particle
counts, same rigid rest shapes -- just translated.  That is what makes
`set_all_positions_from_one()` legal: pushing tile 0's settled particle state
onto every other tile would otherwise be writing positions onto rigid bodies
with mismatched rest configurations, which blows the solver up.

Coordinate frames (the easy thing to get wrong here):
  - Flex world: x right, y up, z depth.  Tile offsets live here as (dx, 0, dz).
  - pybullet: IK and robot bases live here; `transform_bullet_to_flex` maps
    flex_x = bullet_x, flex_y = bullet_z, flex_z = -bullet_y.
    So a flex offset (dx, 0, dz) is a bullet offset (dx, -dz, 0).
  - actions: `action = [sx, sy, ex, ey]` are bullet-frame xy, exactly as in
    FlexEnv.step(), and are given in *tile-local* coordinates here.

Usage:
    env = FlexEnvMulti(config, n_envs=8)
    env.reset()                              # creates + settles all tiles
    p0 = env.get_env_positions(0).copy()     # tile-local particle state
    env.set_all_positions_from_one(p0)       # make every tile identical
    obs_list = env.step(actions)             # actions: (n_envs, 4), tile-local
"""

import sys
sys.path.append('.')

import math

import numpy as np
import pybullet as p
import pyflex

from env.flex_env import FlexEnv, FlexRobotHelper, quatFromAxisAngle
from transformations import quaternion_from_matrix, quaternion_matrix

# Scene id of yx_CarrotsGrid, registered after yx_Coffee_Capsule (23) in
# PyFleX/bindings/pyflex.cpp.
CARROTS_GRID_SCENE_IDX = 24
# Scene id of yx_CoffeeGrid, registered immediately after yx_CarrotsGrid.
COFFEE_GRID_SCENE_IDX = 25
# Scene id of yx_CapsuleGrid, registered immediately after yx_CoffeeGrid.
CAPSULE_GRID_SCENE_IDX = 26

# obj -> (tiled scene index, this env's scene_params builder)
_GRID_SCENES = {
    'carrots': (CARROTS_GRID_SCENE_IDX, '_carrots_scene_params'),
    'coffee': (COFFEE_GRID_SCENE_IDX, '_coffee_scene_params'),
    'capsule': (CAPSULE_GRID_SCENE_IDX, '_capsule_scene_params'),
}


def grid_shape(n_envs):
    """Most-square (nx, nz) tiling with nx * nz >= n_envs."""
    nx = int(math.ceil(math.sqrt(n_envs)))
    nz = int(math.ceil(n_envs / float(nx)))
    return nx, nz


class FlexEnvMulti(FlexEnv):
    def __init__(self, config=None, n_envs=None):
        super().__init__(config)

        par_cfg = (config or {}).get('parallel', {}) or {}
        if n_envs is None:
            n_envs = int(par_cfg.get('n_envs', 4))
        n_envs = max(1, int(n_envs))

        self.grid_nx, self.grid_nz = grid_shape(n_envs)
        # Every tile in the grid is simulated, so rounding the request up to a
        # full rectangle costs nothing extra -- we may as well hand the caller
        # the extra envs rather than paying for tiles it can't use.
        self.n_envs = self.grid_nx * self.grid_nz
        if self.n_envs != n_envs:
            print('FlexEnvMulti: %d envs requested, using %d (%dx%d grid)'
                  % (n_envs, self.n_envs, self.grid_nx, self.grid_nz))

        # One empty scene width between neighbours in each cardinal direction:
        # a tile's container is global_scale wide, so pitch = 2 * global_scale.
        self.tile_pitch = float(par_cfg.get('tile_pitch_scale', 2.0)) * self.global_scale
        self.tile_seed = int(par_cfg.get('tile_seed', 12345))
        self.nan_check_every = int(par_cfg.get('nan_check_every', 1))
        # Draw a frame on every solver step (slow; only useful for a live view).
        self._step_render = bool(par_cfg.get('step_render', False))

        # Flex-frame (x, z) offset of each tile, centered on the origin, in the
        # same order the C++ scene emits them (t_z outer, t_x inner).
        offsets = []
        for t_z in range(self.grid_nz):
            for t_x in range(self.grid_nx):
                off_x = (t_x - 0.5 * (self.grid_nx - 1)) * self.tile_pitch
                off_z = (t_z - 0.5 * (self.grid_nz - 1)) * self.tile_pitch
                offsets.append((off_x, off_z))
        self.env_offsets = np.array(offsets)  # (n_envs, 2): flex (x, z)

        # One helper per env: FlexRobotHelper.loadURDF only creates a pybullet
        # body when its own robotId is None, so distinct helpers give distinct
        # robots (and distinct mesh sets in flex).
        self.flex_robot_helpers = [FlexRobotHelper() for _ in range(self.n_envs)]
        self.robotIds = [None] * self.n_envs
        self.n_particles_per_env = None
        self.n_rigids_per_env = None
        self.last_exploded_envs = []
        self.last_aborted = False

    # ---------------------------------------------------------------- frames

    def env_offset_flex(self, env_idx):
        """(dx, dy, dz) offset of a tile in the flex world frame."""
        off_x, off_z = self.env_offsets[env_idx]
        return np.array([off_x, 0.0, off_z])

    def env_offset_bullet(self, env_idx):
        """(dx, dy, dz) offset of a tile in the pybullet frame."""
        off_x, off_z = self.env_offsets[env_idx]
        return np.array([off_x, -off_z, 0.0])

    # ------------------------------------------------------------ scene init

    def _carrots_scene_params(self):
        """The 20 yx_Carrots scene params for this config's init_pos.

        Copied verbatim from FlexEnv.reset()'s `obj == 'carrots'` branch so the
        tiles are parameterised exactly like the single-env scene.
        """
        if self.obj != 'carrots':
            raise NotImplementedError(
                "_carrots_scene_params is for obj='carrots' only (got %r)."
                % self.obj)

        staticFriction = 1.0
        dynamicFriction = 0.9
        draw_skin = 1.0
        min_dist = 10.0
        max_dist = 20.0
        self.cvx_region = np.zeros((1,4)) # every row: left, right, bottom, top
        self.cvx_region[0,0] = -self.wkspc_w
        self.cvx_region[0,1] = self.wkspc_w
        self.cvx_region[0,2] = -self.wkspc_w
        self.cvx_region[0,3] = self.wkspc_w
        if self.init_pos == 'count_target':
            # Directly targets an exact carrot count (`self.target_num_carrots`,
            # set by the caller before reset()), unlike every other init_pos
            # variant here, which derives count AS A SIDE EFFECT of a randomly
            # drawn blob_r/scale -- measured empirically, that gives only 5
            # distinct count values for rand_blob (45/105/189/297/429) and 5
            # for rand_spread (189/297/429/585/765), none in e.g. [10,30] or
            # [50,70]. `num_objects_override` alone isn't a fix either: it just
            # truncates whatever footprint rand_blob/rand_spread happened to
            # compute, which for a small target count on a large footprint
            # produces a thin sliver/corner of carrots instead of a natural
            # compact pile.
            #
            # Fix: invert rand_blob's own count formula,
            # num_carrots = (num_x * num_z - 1) * 3 with num_x == num_z,
            # for num_x from the TARGET count instead of from a random blob_r,
            # so the footprint is sized to the target directly (no truncation
            # waste) while keeping the same "a few stacked layers" pile shape
            # and the same per-carrot scale/position-offset randomization as
            # rand_blob for in-group state variety.
            target = int(self.target_num_carrots)
            if target < 1:
                raise ValueError(
                    'target_num_carrots must be >= 1 (got %d)' % target)
            rand_scale = np.random.uniform(0.07, 0.12) * self.global_scale / 8.0
            max_scale = rand_scale
            min_scale = rand_scale
            num_x = max(2, int(np.ceil(np.sqrt(target / 3.0 + 1.0))))
            num_z = num_x
            num_y = 10
            inter_space = max_scale
            x = -0.5 * num_x * max_scale
            y = 0.5
            z = -0.5 * num_z * max_scale
            x_off = self.global_scale * np.random.uniform(-1./12., 1./8.)
            z_off = self.global_scale * np.random.uniform(-1./12., 1./8.)
            x += x_off
            z += z_off
            num_carrots = target
            add_singular = 0.0
            add_sing_x = -1
            add_sing_y = -1
            add_sing_z = -1
            add_noise = 0.0
        elif self.init_pos == 'spread':
            max_scale = 0.1 * self.global_scale / 8.0
            min_scale = 0.1 * self.global_scale / 8.0
            x = -1.5 * self.global_scale / 8.0 # -0.35; -1.2
            y = 0.5
            z = -1.5 * self.global_scale / 8.0 # -0.35; -1.2
            inter_space = 2 * max_scale
            num_x = int(abs(x/2.) / max_scale + 1) * 2 + 1
            num_y = 10
            num_z = int(abs(z/2.) / max_scale + 1) * 2 + 1
            num_carrots = (num_x * num_z - 1) * 3
            add_singular = 0.0
            add_sing_x = -1
            add_sing_y = -1
            add_sing_z = -1
            add_noise = 0.0
        elif self.init_pos == 'wkspc_spread':
            max_scale = 0.2 * self.global_scale / 8.0
            min_scale = 0.2 * self.global_scale / 8.0
            x = -1.2 * self.global_scale / 8.0 # -0.35; -1.2
            y = 0.5
            z = -1.2 * self.global_scale / 8.0 # -0.35; -1.2
            inter_space = 2 * max_scale
            num_x = int(abs(x/2.) / max_scale + 1) * 2
            num_y = 10
            num_z = int(abs(z/2.) / max_scale + 1) * 2
            num_carrots = num_x * num_z - 1
            add_singular = 0.0
            add_sing_x = -1
            add_sing_y = -1
            add_sing_z = -1
            add_noise = 0.0
        elif self.init_pos == 'wkspc_spread_double':
            max_scale = 0.2 * self.global_scale / 8.0
            min_scale = 0.2 * self.global_scale / 8.0
            x = -1.2 * self.global_scale / 8.0 # -0.35; -1.2
            y = 0.5
            z = -1.2 * self.global_scale / 8.0 # -0.35; -1.2
            inter_space = 2 * max_scale
            num_x = int(abs(x/2.) / max_scale + 1) * 2
            num_y = 10
            num_z = int(abs(z/2.) / max_scale + 1) * 2
            num_carrots = 2 * (num_x * num_z - 1)
            add_singular = 0.0
            add_sing_x = -1
            add_sing_y = -1
            add_sing_z = -1
            add_noise = 0.0
        elif self.init_pos == 'wkspc_spread_triple':
            max_scale = 0.2 * self.global_scale / 8.0
            min_scale = 0.2 * self.global_scale / 8.0
            x = -1.2 * self.global_scale / 8.0 # -0.35; -1.2
            y = 0.5
            z = -1.2 * self.global_scale / 8.0 # -0.35; -1.2
            inter_space = 2 * max_scale
            num_x = int(abs(x/2.) / max_scale + 1) * 2
            num_y = 10
            num_z = int(abs(z/2.) / max_scale + 1) * 2
            num_carrots = 3 * (num_x * num_z - 1)
            add_singular = 0.0
            add_sing_x = -1
            add_sing_y = -1
            add_sing_z = -1
            add_noise = 0.0
        elif self.init_pos == 'wkspc_spread_4':
            max_scale = 0.2 * self.global_scale / 8.0
            min_scale = 0.2 * self.global_scale / 8.0
            x = -1.2 * self.global_scale / 8.0 # -0.35; -1.2
            y = 0.5
            z = -1.2 * self.global_scale / 8.0 # -0.35; -1.2
            inter_space = 2 * max_scale
            num_x = int(abs(x/2.) / max_scale + 1) * 2
            num_y = 10
            num_z = int(abs(z/2.) / max_scale + 1) * 2
            num_carrots = 4 * (num_x * num_z - 1)
            add_singular = 0.0
            add_sing_x = -1
            add_sing_y = -1
            add_sing_z = -1
            add_noise = 0.0
        elif self.init_pos == 'extra_large_wkspc_spread':
            max_scale = 0.3 * self.global_scale / 8.0
            min_scale = 0.3 * self.global_scale / 8.0
            x = -1.2 * self.global_scale / 8.0 # -0.35; -1.2
            y = 0.5
            z = -1.2 * self.global_scale / 8.0 # -0.35; -1.2
            inter_space = 2 * max_scale
            num_x = int(abs(x/2.) / max_scale) * 2
            num_y = 10
            num_z = int(abs(z/2.) / max_scale) * 2
            num_carrots = 2 * (num_x * num_z - 1)
            add_singular = 0.0
            add_sing_x = -1
            add_sing_y = -1
            add_sing_z = -1
            add_noise = 0.0
        elif self.init_pos == 'extra_small_wkspc_spread':
            max_scale = 0.09 * self.global_scale / 8.0
            min_scale = 0.09 * self.global_scale / 8.0
            x = -1.2 * self.global_scale / 8.0 # -0.35; -1.2
            y = 0.5
            z = -1.2 * self.global_scale / 8.0 # -0.35; -1.2
            inter_space = 2 * max_scale
            num_x = int(abs(x/2.) / max_scale + 1) * 2
            num_y = 10
            num_z = int(abs(z/2.) / max_scale + 1) * 2
            num_carrots = 4 * (num_x * num_z - 1)
            add_singular = 0.0
            add_sing_x = -1
            add_sing_y = -1
            add_sing_z = -1
            add_noise = 0.0
        elif self.init_pos == 'extra_small_half_spread':
            max_scale = 0.09 * self.global_scale / 8.0
            min_scale = 0.09 * self.global_scale / 8.0
            x = -0.9 * self.global_scale / 8.0 # -0.35; -1.2
            y = 0.5
            z = -0.9 * self.global_scale / 8.0 # -0.35; -1.2
            inter_space = 2 * max_scale
            num_x = int(abs(x/2.) / max_scale + 1) * 2
            num_y = 10
            num_z = int(abs(z/2.) / max_scale + 1) * 2
            num_carrots = 4 * (num_x * num_z - 1)
            add_singular = 0.0
            add_sing_x = -1
            add_sing_y = -1
            add_sing_z = -1
            add_noise = 0.0
        elif self.init_pos == 'rand_blob':
            rand_scale = np.random.uniform(0.07, 0.12) * self.global_scale / 8.0
            max_scale = rand_scale
            min_scale = rand_scale
            blob_r = np.random.uniform(0.3, 0.5)
            x = -blob_r * self.global_scale / 8.0
            y = 0.5
            z = -blob_r * self.global_scale / 8.0
            inter_space = max_scale
            num_x = int(abs(x) / max_scale) * 2
            num_y = 10
            num_z = int(abs(z) / max_scale) * 2
            x_off = self.global_scale * np.random.uniform(-1./12., 1./8.)
            z_off = self.global_scale * np.random.uniform(-1./12., 1./8.)
            x += x_off
            z += z_off
            print('rand_scale: ', rand_scale)
            print('blob_r: ', blob_r)
            print('x_off: ', x_off)
            print('z_off: ', z_off)
            num_carrots = (num_x * num_z - 1) * 3
            add_singular = 0.0
            add_sing_x = -1
            add_sing_y = -1
            add_sing_z = -1
            add_noise = 0.0
        elif self.init_pos == 'rand_spread':
            rand_scale = np.random.uniform(0.09, 0.12) * self.global_scale / 8.0
            max_scale = rand_scale
            min_scale = rand_scale
            blob_r = np.random.uniform(0.7, 1.0)
            x = - blob_r * self.global_scale / 8.0
            y = 0.5
            z = - blob_r * self.global_scale / 8.0
            inter_space = 1.5 * max_scale
            num_x = int(abs(x/1.5) / max_scale + 1) * 2
            num_y = 10
            num_z = int(abs(z/1.5) / max_scale + 1) * 2
            x_off = self.global_scale * np.random.uniform(-1./24., 1./24.)
            z_off = self.global_scale * np.random.uniform(-1./24., 1./24.)
            x += x_off
            z += z_off
            num_carrots = (num_x * num_z - 1) * 3
            add_singular = 0.0
            add_sing_x = -1
            add_sing_y = -1
            add_sing_z = -1
            add_noise = 0.0
        elif self.init_pos == 'rand_sparse_spread':
            rand_scale = 0.12 * self.global_scale / 8.0
            max_scale = rand_scale
            min_scale = rand_scale
            blob_r = np.random.uniform(1.0, 1.5)
            x = - blob_r * self.global_scale / 8.0
            y = 0.5
            z = - blob_r * self.global_scale / 8.0
            inter_space = max_scale * 2
            num_x = int(abs(x/2.) / max_scale) * 2
            num_y = 10
            num_z = int(abs(z/2.) / max_scale) * 2
            # x_off = self.global_scale * np.random.uniform(-1./24., 1./24.)
            # z_off = self.global_scale * np.random.uniform(-1./24., 1./24.)
            # x += x_off
            # z += z_off
            num_carrots = (num_x * num_z - 1) * 1
            add_singular = 0.0
            add_sing_x = -1
            add_sing_y = -1
            add_sing_z = -1
            add_noise = 0.0
        elif self.init_pos == 'rb_corner':
            max_scale = 0.12 * self.global_scale / 8.0
            min_scale = 0.12 * self.global_scale / 8.0
            x = -0.4 * self.global_scale / 8.0 # -0.35; -1.2
            y = 0.5
            z = -0.4 * self.global_scale / 8.0 # -0.35; -1.2
            inter_space = max_scale
            num_x = int(abs(x) / max_scale) * 2
            num_y = 10
            num_z = int(abs(z) / max_scale) * 2
            x += self.global_scale / 8.
            z += self.global_scale / 8.
            num_carrots = (num_x * num_z - 1) * 3
            add_singular = 0.0
            add_sing_x = -1
            add_sing_y = -1
            add_sing_z = -1
            add_noise = 0.0
        elif self.init_pos == 'center':
            max_scale = 0.12 * self.global_scale / 8.0
            min_scale = 0.12 * self.global_scale / 8.0
            x = -0.4 * self.global_scale / 8.0 # -0.35; -1.2
            y = 0.5
            z = -0.4 * self.global_scale / 8.0 # -0.35; -1.2
            inter_space = max_scale
            num_x = int(abs(x) / max_scale) * 2
            num_y = 10
            num_z = int(abs(z) / max_scale) * 2
            num_carrots = (num_x * num_z - 1) * 3
            add_singular = 0.0
            add_sing_x = -1
            add_sing_y = -1
            add_sing_z = -1
            add_noise = 0.0
        elif self.init_pos == 'center_init_2':
            max_scale = 0.12 * self.global_scale / 8.0
            min_scale = 0.12 * self.global_scale / 8.0
            x = -1.0 * self.global_scale / 8.0 # -0.35; -1.2
            y = 0.5
            z = -1.0 * self.global_scale / 8.0 # -0.35; -1.2
            inter_space = max_scale * 2
            num_x = int(abs(x/2.) / max_scale) * 2
            num_y = 10
            num_z = int(abs(z/2.) / max_scale) * 2
            num_carrots = (num_x * num_z - 1) * 1
            add_singular = 0.0
            add_sing_x = -1
            add_sing_y = -1
            add_sing_z = -1
            add_noise = 1.0
        elif self.init_pos == 'rt_corner':
            max_scale = 0.15 * self.global_scale / 8.0
            min_scale = 0.15 * self.global_scale / 8.0
            x = -0.35 * self.global_scale / 8.0 # -0.35; -1.2
            y = 0.5
            z = -0.35 * self.global_scale / 8.0 # -0.35; -1.2
            inter_space = max_scale
            num_x = int(abs(x) / max_scale) * 2
            num_y = 10
            num_z = int(abs(z) / max_scale) * 2
            x += self.global_scale / 8.
            z -= self.global_scale / 8.
            num_carrots = int(0.25 * self.global_scale / (max_scale ** 2))
            add_singular = 0.0
            add_sing_x = -1
            add_sing_y = -1
            add_sing_z = -1
            add_noise = 0.0
        elif self.init_pos == 'wkspc_spread_multi_granularity':
            max_scale = 0.2 * self.global_scale / 8.0
            min_scale = 0.05 * self.global_scale / 8.0
            x = -1.2 * self.global_scale / 8.0 # -0.35; -1.2
            y = 0.5
            z = -1.2 * self.global_scale / 8.0 # -0.35; -1.2
            inter_space = 2 * max_scale
            num_x = int(abs(x/2.) / max_scale + 1) * 2
            num_y = 10
            num_z = int(abs(z/2.) / max_scale + 1) * 2
            num_carrots = (num_x * num_z - 1) * 2
            add_singular = 0.0
            add_sing_x = -1
            add_sing_y = -1
            add_sing_z = -1
            add_noise = 0.0
        elif self.init_pos == 'singular':
            max_scale = 0.15 * self.global_scale / 8.0
            min_scale = 0.15 * self.global_scale / 8.0
            x = -0.35 * self.global_scale / 8.0 # -0.35; -1.2
            y = 0.5
            z = -0.35 * self.global_scale / 8.0 # -0.35; -1.2
            inter_space = max_scale
            num_x = int(abs(x) / max_scale) * 2
            num_y = 10
            num_z = int(abs(z) / max_scale) * 2
            x -= self.global_scale / 8.
            num_carrots = int(0.25 * self.global_scale / (max_scale ** 2))
            add_singular = 1.0
            add_sing_x = 3.0 * self.global_scale / 24.0
            add_sing_y = 0.5
            add_sing_z = 0.0
            add_noise = 0.0
        elif self.init_pos == 'blank':
            max_scale = 0.15 * self.global_scale / 8.0
            min_scale = 0.15 * self.global_scale / 8.0
            x = -0.35 * self.global_scale / 8.0 # -0.35; -1.2
            y = 0.5
            z = -0.35 * self.global_scale / 8.0 # -0.35; -1.2
            inter_space = max_scale
            num_x = 1
            num_y = 10
            num_z = 1
            x -= self.global_scale
            num_carrots = 1
            add_singular = 0.0
            add_sing_x = 3.0 * self.global_scale / 24.0
            add_sing_y = 0.5
            add_sing_z = 0.0
            add_noise = 0.0
        else:
            raise NotImplementedError

        scene_params = np.array([max_scale,
                                 min_scale,
                                 x,
                                 y,
                                 z,
                                 staticFriction,
                                 dynamicFriction,
                                 draw_skin,
                                 num_carrots,
                                 min_dist,
                                 max_dist,
                                 num_x,
                                 num_y,
                                 num_z,
                                 inter_space,
                                 add_singular,
                                 add_sing_x,
                                 add_sing_y,
                                 add_sing_z,
                                 add_noise])
        if self.num_objects_override is not None:
            scene_params[8] = int(self.num_objects_override)
        return scene_params

    def _coffee_scene_params(self):
        """The 8 yx_Coffee scene params (copied verbatim from FlexEnv.reset()'s
        `obj == 'coffee'` branch, same as single-env coffee; unlike carrots,
        coffee's generation does not depend on `init_pos`), plus the two
        params yx_CoffeeGrid adds beyond yx_Coffee: `particle_radius` (packing
        resolution -- see yx_coffee_grid.h) and `pos_noise` (per-bean jitter,
        giving per-state variety the way carrots' rand_blob/rand_spread do).

        `particle_radius` defaults to `dataset.particle_r` (falling back to
        coffee's original hardcoded 0.075) so a dataset can trade bean
        resolution for particle count/sim cost; `pos_noise` is half the bean
        pitch, the same ratio yx_CarrotsGrid uses for its own position noise.
        """
        if self.obj != 'coffee':
            raise NotImplementedError(
                "_coffee_scene_params is for obj='coffee' only (got %r)."
                % self.obj)

        scale = 0.2 * self.global_scale / 8.0
        x = -0.9 * self.global_scale / 8.0
        y = 0.5
        z = -0.9 * self.global_scale / 8.0
        staticFriction = 0.0
        dynamicFriction = 1.0
        draw_skin = 1.0
        num_coffee = 1000  # [200, 1000]
        if self.num_objects_override is not None:
            num_coffee = int(self.num_objects_override)
        particle_radius = float(self.config['dataset'].get('particle_r', 0.075))
        pos_noise = 0.5 * scale
        return np.array([scale, x, y, z, staticFriction, dynamicFriction,
                          draw_skin, num_coffee, particle_radius, pos_noise])

    def _capsule_scene_params(self):
        """The 10 yx_Capsule scene params (copied verbatim from FlexEnv.reset()'s
        `obj == 'capsule'` branch, same as single-env capsule; like coffee,
        capsule's generation does not depend on `init_pos`), plus the two
        params yx_CapsuleGrid adds beyond yx_Capsule: `particle_radius`
        (packing resolution -- see yx_capsule_grid.h) and `pos_noise`
        (per-capsule jitter, giving per-state variety the way carrots'
        rand_blob/rand_spread do).

        `particle_radius` defaults to `dataset.particle_r` (falling back to
        capsule's original hardcoded 0.075) so a dataset can trade capsule
        resolution for particle count/sim cost; `pos_noise` is half the
        capsule pitch, the same ratio yx_CoffeeGrid / yx_CarrotsGrid use.
        """
        if self.obj != 'capsule':
            raise NotImplementedError(
                "_capsule_scene_params is for obj='capsule' only (got %r)."
                % self.obj)

        scale = 0.2 * self.global_scale / 8.0
        x = -1. * self.global_scale / 8.0
        y = 0.5
        z = -1. * self.global_scale / 8.0
        staticFriction = 0.0
        dynamicFriction = 0.5
        draw_skin = 1.0
        num_capsule = 200  # [200, 1000]
        if self.num_objects_override is not None:
            num_capsule = int(self.num_objects_override)
        slices = 10
        segments = 20
        particle_radius = float(self.config['dataset'].get('particle_r', 0.075))
        pos_noise = 0.5 * scale
        return np.array([scale, x, y, z, staticFriction, dynamicFriction,
                          draw_skin, num_capsule, slices, segments,
                          particle_radius, pos_noise])

    def reset(self, tile_seed=None):
        """Build and settle the grid.

        `tile_seed` selects which pile is generated. It is shared by all tiles
        (that is what makes them identical), so it must be varied between
        *states*: calling reset() with the same seed regenerates the very same
        pile, which for a data collection run would mean every 'initial state'
        is the same state. The collector passes a per-state seed.
        """
        if tile_seed is not None:
            self.tile_seed = int(tile_seed)
        if self.obj not in _GRID_SCENES:
            raise NotImplementedError(
                "FlexEnvMulti supports obj in %r (got %r); each needs its own "
                "tiled scene (see yx_CarrotsGrid / yx_CoffeeGrid / yx_CapsuleGrid)."
                % (sorted(_GRID_SCENES), self.obj))
        scene_idx, params_fn_name = _GRID_SCENES[self.obj]
        base_params = getattr(self, params_fn_name)()
        scene_params = np.concatenate([
            base_params,
            np.array([self.grid_nx,
                      self.grid_nz,
                      self.tile_pitch,
                      self.tile_pitch,
                      self.tile_seed], dtype=np.float64),
        ]).astype(np.float64)
        self.scene_params = scene_params
        pyflex.set_scene(scene_idx, scene_params, 0)

        pyflex.set_camPos(self.camPos)
        pyflex.set_camAngle(self.camAngle)

        n_particles = self.get_positions().reshape(-1, 4).shape[0]
        if n_particles % self.n_envs != 0:
            raise RuntimeError(
                'Tiled scene produced %d particles, not divisible by %d envs -- '
                'the tiles are not identical, so per-env slicing would be wrong.'
                % (n_particles, self.n_envs))
        self.n_particles_per_env = n_particles // self.n_envs

        n_rigids = pyflex.get_n_rigids()
        if n_rigids % self.n_envs != 0:
            raise RuntimeError(
                'Tiled scene produced %d rigid bodies, not divisible by %d envs.'
                % (n_rigids, self.n_envs))
        self.n_rigids_per_env = n_rigids // self.n_envs

        self.check_tile_generation()

        for _ in range(self.reset_warmup_steps):
            self._sim_step()

        # --- walls: one container per tile ---
        halfEdge = np.array([0.05, 1.0, self.global_scale / 2.0])
        base_centers = [np.array([self.global_scale / 2.0, 1.0, 0.0]),
                        np.array([0.0, 1.0, -self.global_scale / 2.0]),
                        np.array([-self.global_scale / 2.0, 1.0, 0.0]),
                        np.array([0.0, 1.0, self.global_scale / 2.0])]
        quats = [quatFromAxisAngle(axis=np.array([0., 1., 0.]), angle=0.),
                 quatFromAxisAngle(axis=np.array([0., 1., 0.]), angle=np.pi / 2.),
                 quatFromAxisAngle(axis=np.array([0., 1., 0.]), angle=0.),
                 quatFromAxisAngle(axis=np.array([0., 1., 0.]), angle=np.pi / 2.)]
        hideShape = 0
        color = np.ones(3) * 0.9
        self.wall_shape_states = np.zeros((4 * self.n_envs, 14))
        for env_idx in range(self.n_envs):
            off = self.env_offset_flex(env_idx)
            for i, base_center in enumerate(base_centers):
                center = base_center + off
                pyflex.add_box(halfEdge, center, quats[i], hideShape, color)
                self.wall_shape_states[env_idx * 4 + i] = np.concatenate(
                    [center, center, quats[i], quats[i]])

        # --- robots: one per tile ---
        if self.robot_type != 'franka':
            raise NotImplementedError(
                "FlexEnvMulti supports robot_type='franka' only (got %r)." % self.robot_type)
        self.rest_joints = [np.pi * 5 / 8, -np.pi / 2, -np.pi / 2, -np.pi * 5 / 8,
                            -np.pi / 4, np.pi / 2, np.pi / 4, 0., 0.]
        base_pos = np.array([-4.5 * self.global_scale / 8.0, 0, 0])
        for env_idx in range(self.n_envs):
            helper = self.flex_robot_helpers[env_idx]
            self.robotIds[env_idx] = helper.loadURDF(
                self.franka_urdf,
                (base_pos + self.env_offset_bullet(env_idx)).tolist(),
                [0, 0, 0, 1],
                globalScaling=self.global_scale)

        self.num_joints = p.getNumJoints(self.robotIds[0])
        self.joints_lower = np.zeros(self.num_dofs)
        self.joints_upper = np.zeros(self.num_dofs)
        # Movable joint indices are fixed for the whole run; the original code
        # re-queried getJointInfo for every joint on every reset_panda() call,
        # i.e. once per joint per env per solver step.
        self.movable_joints = []
        dof_idx = 0
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robotIds[0], i)
            jointType = info[2]
            if jointType in (p.JOINT_PRISMATIC, p.JOINT_REVOLUTE):
                self.joints_lower[dof_idx] = info[8]
                self.joints_upper[dof_idx] = info[9]
                self.movable_joints.append(i)
                dof_idx += 1

        # Damping is a one-off per-body property; the original code re-applied
        # it on every reset_panda() call, which is pure overhead in the inner
        # loop and matters more here with N robots.
        for robotId in self.robotIds:
            for j in range(self.num_joints):
                p.changeDynamics(robotId, j, linearDamping=0, angularDamping=0)

        self.last_ee = None
        # Park the arm in the SAME neutral pose that step() retracts to before
        # capturing an after-image. Leaving it at rest_joints (the old
        # behaviour) folds it out over the workspace, so every initial_color.png
        # came out with the arm covering most of the pile while the after-images
        # were clear -- the two were not comparable. The rest pose also hovers
        # directly above material that settled before the robot was loaded, so
        # the render step could let it intrude into the pile. Applied twice: the
        # shape states carry a previous pose for continuous collision, and a
        # single call would sweep the arm through the scene to get here.
        self.reset_pandas([np.zeros(9).tolist()] * self.n_envs)
        self.reset_pandas([np.zeros(9).tolist()] * self.n_envs)

        # Settling is chaotic: identical tiles integrated in one solver still
        # drift apart by ~O(particle radius) over the warmup, because neighbour
        # ordering and memory layout differ per tile.  So rather than trusting
        # the warmup to keep them equal, stamp tile 0's settled state onto every
        # tile -- valid precisely because the bodies are identical (verified
        # pre-warmup by check_tile_generation), and it leaves every env at a
        # genuinely settled configuration.
        self.synchronize_tiles()

    def check_tile_generation(self, tol=1e-4):
        """Verify tiles were *generated* identically, before any dynamics run.

        This is the assumption the whole batching scheme rests on: tile k must
        be tile 0 body-for-body (same particle count, same rigid rest shape),
        or writing tile 0's state onto tile k is writing onto bodies whose
        shape-matching rest configs disagree, and the solver explodes.  Checked
        at spawn time, where the tiles should agree to floating-point noise --
        after any simulation they legitimately diverge.
        """
        if self.n_envs == 1:
            return 0.0
        pos = self.get_positions().reshape(-1, 4)
        ref = pos[self._env_slice(0)][:, :3] - self.env_offset_flex(0)
        max_dev = 0.0
        for env_idx in range(1, self.n_envs):
            other = pos[self._env_slice(env_idx)][:, :3] - self.env_offset_flex(env_idx)
            max_dev = max(max_dev, float(np.abs(other - ref).max()))
        if max_dev > tol:
            raise RuntimeError(
                'Tiles were not generated identically (max deviation %.6g at spawn). '
                'The per-tile RNG reset in yx_CarrotsGrid is not taking effect -- '
                'check that the compiled pyflex is the rebuilt one.' % max_dev)
        return max_dev

    def synchronize_tiles(self, env_idx=0):
        """Make every tile an exact copy of `env_idx`, at rest."""
        state = self.get_env_state(env_idx)
        state['velocities'][:] = 0.0
        self.set_all_envs_state(state)

    # ----------------------------------------------------------- robot state

    def _robot_shape_states(self, env_idx):
        """Flex shape states for one robot, using batched pybullet queries.

        Same output as FlexRobotHelper.getRobotShapeStates(), but that version
        issues one getLinkState call per link -- ~12 pybullet round trips per
        env per solver step, which is a few hundred thousand calls over a batch
        and is a real share of the runtime here. getLinkStates fetches them in
        one call.
        """
        helper = self.flex_robot_helpers[env_idx]
        robotId = self.robotIds[env_idx]

        base_com_pos, base_com_orn = p.getBasePositionAndOrientation(robotId)
        di = p.getDynamicsInfo(robotId, -1)
        pos_inv, orn_inv = p.invertTransform(di[3], di[4])
        pos, orn = p.multiplyTransforms(base_com_pos, base_com_orn, pos_inv, orn_inv)

        state_cur = np.empty((helper.num_link, 8))
        state_cur[0] = list(pos) + [1] + list(orn)
        link_states = p.getLinkStates(robotId, list(range(helper.num_link - 1)))
        for l, ls in enumerate(link_states):
            state_cur[l + 1] = list(ls[4]) + [1] + list(ls[5])

        if helper.state_pre is None:
            helper.state_pre = state_cur.copy()
        state_pre = helper.state_pre

        T = helper.transform_bullet_to_flex
        shape_states = np.zeros((helper.num_meshes, 14))
        mesh_idx = 0
        for i in range(helper.num_link):
            if not helper.has_mesh[i]:
                continue
            shape_states[mesh_idx, 0:3] = np.matmul(T, state_cur[i, :4])[:3]
            shape_states[mesh_idx, 3:6] = np.matmul(T, state_pre[i, :4])[:3]
            shape_states[mesh_idx, 6:10] = quaternion_from_matrix(
                np.matmul(T, quaternion_matrix(state_cur[i, 4:])))
            shape_states[mesh_idx, 10:14] = quaternion_from_matrix(
                np.matmul(T, quaternion_matrix(state_pre[i, 4:])))
            mesh_idx += 1

        helper.state_pre = state_cur
        return shape_states

    def _shape_states(self):
        """Full shape-state array: all walls, then each env's robot meshes.

        Order must match shape creation order (all boxes were added before any
        robot mesh), because pyflex.set_shape_states() writes one flat buffer.
        """
        robot_states = [self._robot_shape_states(i) for i in range(self.n_envs)]
        return np.concatenate([self.wall_shape_states] + robot_states, axis=0)

    def reset_pandas(self, joint_positions_per_env):
        """Set every robot's joints, then push all shape states in one call."""
        for env_idx, jointPositions in enumerate(joint_positions_per_env):
            robotId = self.robotIds[env_idx]
            for index, j in enumerate(self.movable_joints):
                # Mirrors FlexEnv.reset_panda: the two finger joints are
                # always held at 0 rather than tracking the IK solution.
                pose = jointPositions[index] if index < self.num_dofs - 2 else 0.
                p.resetJointState(robotId, j, pose)
        pyflex.set_shape_states(self._shape_states())

    # ------------------------------------------------------------- particles

    def _env_slice(self, env_idx):
        n = self.n_particles_per_env
        return slice(env_idx * n, (env_idx + 1) * n)

    def get_env_positions(self, env_idx):
        """Tile-local (x, y, z, invmass) particle state of one env."""
        pos = self.get_positions().reshape(-1, 4)[self._env_slice(env_idx)].copy()
        pos[:, :3] -= self.env_offset_flex(env_idx)
        return pos.reshape(-1)

    def get_all_env_positions(self):
        return [self.get_env_positions(i) for i in range(self.n_envs)]

    def _sim_step(self):
        """Advance the solver without drawing a frame.

        pyflex.step() renders and presents a full frame on every call, which
        measured ~8x the cost of the physics itself (11.4 vs 1.4 ms/step at 20k
        particles). Data collection only needs an image at the end of a push,
        so stepping skips the draw and rendering happens in render_env().
        Falls back to a drawing step on a pyflex built without the flag.
        """
        if self._step_render:
            pyflex.step()
            return
        try:
            pyflex.step(render=0)
        except TypeError:
            # Older pyflex build without the `render` argument.
            self._step_render = True
            print('FlexEnvMulti: pyflex.step() has no render flag; rebuild '
                  'PyFleX to get the ~8x faster non-drawing step.')
            pyflex.step()

    def sample_action_obj_biased_local(self, n, env_idx=0, rng=None):
        """Start points drawn near the pile, end points uniform.

        FlexEnv.sample_action_obj_biased() cannot be used here for two reasons:
        it reads get_positions(), which spans every tile, and it clips to
        +/-wkspc_w while ignoring action_margin -- which would put the plate
        right against the container walls, the very thing the margin exists to
        prevent. This version works off one tile's particles and respects the
        margin on both the start and the end point.

        Action coordinates are bullet-frame (x, -z), matching step().
        """
        rng = rng if rng is not None else np.random
        lo = -self.wkspc_w + self.action_margin
        hi = self.wkspc_w - self.action_margin
        pos = self.get_env_positions(env_idx).reshape(-1, 4)
        idx = rng.randint(0, pos.shape[0], size=n)
        start = np.stack([pos[idx, 0], -pos[idx, 2]], axis=1)
        sigma = 0.5 * self.global_scale / 12.0
        start = start + rng.normal(0, sigma, size=start.shape)
        actions = np.zeros((n, 4))
        actions[:, :2] = np.clip(start, lo, hi)
        actions[:, 2:] = lo + (hi - lo) * rng.rand(n, 2)
        return actions

    def _rigid_slice(self, env_idx):
        n = self.n_rigids_per_env
        return slice(env_idx * n, (env_idx + 1) * n)

    def get_env_state(self, env_idx):
        """Complete tile-local state of one env: particles + rigid frames.

        Particle positions alone are not a full state for a rigid-body scene.
        The solver carries each body's shape-matching frame (rotation + center
        of mass) between steps, so a restore that sets only positions leaves
        stale frames behind and the next step snaps the bodies somewhere else.
        Observed directly here: tiles synced to a bit-identical particle state
        diverged by 0.25 world units on the very first step until the frames
        were restored too.
        """
        pos = self.get_positions().reshape(-1, 4)[self._env_slice(env_idx)].copy()
        pos[:, :3] -= self.env_offset_flex(env_idx)
        vel = self.get_velocities().reshape(-1, 3)[self._env_slice(env_idx)].copy()
        rot = pyflex.get_rigidRotations().reshape(-1, 4)[self._rigid_slice(env_idx)].copy()
        trans = pyflex.get_rigidTranslations().reshape(-1, 3)[self._rigid_slice(env_idx)].copy()
        trans -= self.env_offset_flex(env_idx)
        return {'positions': pos, 'velocities': vel,
                'rigid_rotations': rot, 'rigid_translations': trans}

    def set_all_envs_state(self, state):
        """Broadcast one tile-local state onto every tile.

        Legal only because the tiles are body-for-body identical (see module
        docstring and check_tile_generation); the tile offset is applied to the
        translational quantities, while rotations are offset-invariant.
        """
        pos = np.asarray(state['positions']).reshape(-1, 4)
        vel = np.asarray(state['velocities']).reshape(-1, 3)
        rot = np.asarray(state['rigid_rotations']).reshape(-1, 4)
        trans = np.asarray(state['rigid_translations']).reshape(-1, 3)
        if pos.shape[0] != self.n_particles_per_env:
            raise ValueError('expected %d particles, got %d'
                             % (self.n_particles_per_env, pos.shape[0]))
        if rot.shape[0] != self.n_rigids_per_env:
            raise ValueError('expected %d rigid bodies, got %d'
                             % (self.n_rigids_per_env, rot.shape[0]))

        full_pos = np.zeros((self.n_particles_per_env * self.n_envs, 4), dtype=np.float32)
        full_vel = np.zeros((self.n_particles_per_env * self.n_envs, 3), dtype=np.float32)
        full_rot = np.zeros((self.n_rigids_per_env * self.n_envs, 4), dtype=np.float32)
        full_trans = np.zeros((self.n_rigids_per_env * self.n_envs, 3), dtype=np.float32)
        for env_idx in range(self.n_envs):
            off = self.env_offset_flex(env_idx)
            block = pos.copy()
            block[:, :3] += off
            full_pos[self._env_slice(env_idx)] = block
            full_vel[self._env_slice(env_idx)] = vel
            full_rot[self._rigid_slice(env_idx)] = rot
            full_trans[self._rigid_slice(env_idx)] = trans + off

        pyflex.set_positions(full_pos.reshape(-1))
        pyflex.set_velocities(full_vel.reshape(-1))
        pyflex.set_rigidRotations(full_rot.reshape(-1))
        pyflex.set_rigidTranslations(full_trans.reshape(-1))

    def set_all_positions_from_one(self, local_positions):
        """Positions-only broadcast; prefer set_all_envs_state for a real restore."""
        local = np.asarray(local_positions).reshape(-1, 4)
        if local.shape[0] != self.n_particles_per_env:
            raise ValueError('expected %d particles, got %d'
                             % (self.n_particles_per_env, local.shape[0]))
        full = np.zeros((self.n_particles_per_env * self.n_envs, 4), dtype=np.float32)
        for env_idx in range(self.n_envs):
            block = local.copy()
            block[:, :3] += self.env_offset_flex(env_idx)
            full[self._env_slice(env_idx)] = block
        pyflex.set_positions(full.reshape(-1))

    def zero_velocities(self):
        pyflex.set_velocities(np.zeros_like(self.get_velocities()))

    def env_has_nan(self, env_idx):
        pos = self.get_positions().reshape(-1, 4)[self._env_slice(env_idx)]
        return bool(np.isnan(pos[:, :3]).any())

    # ---------------------------------------------------------------- render

    def render_env(self, env_idx, step_first=False):
        """Render one tile with the same camera geometry as the single-env setup.

        The camera is translated to the tile instead of zooming out over the
        whole grid, so saved images are pixel-comparable to FlexEnv renders.
        """
        if step_first and self.render_step_before_capture:
            self._sim_step()
        pyflex.set_camPos(self.camPos + self.env_offset_flex(env_idx))
        pyflex.set_camAngle(self.camAngle)
        out = pyflex.render(render_depth=True)
        return self._process_render_output(out)

    def render_all(self):
        if self.render_step_before_capture:
            self._sim_step()
        obs = [self.render_env(i) for i in range(self.n_envs)]
        pyflex.set_camPos(self.camPos)
        pyflex.set_camAngle(self.camAngle)
        return obs

    # ------------------------------------------------------------------ step

    def _waypoints(self, action, env_idx):
        """Bullet-frame end-effector waypoints for one env's push."""
        # Push height tracks the live pile height, as in FlexEnv.step(); it is
        # computed per env so a tile whose pile has been flattened by an
        # earlier push still gets a sane height.
        pos = self.get_positions().reshape(-1, 4)[self._env_slice(env_idx)]
        h = 0.5 * float(np.nanmax(pos[:, 1]))
        off = self.env_offset_bullet(env_idx)
        s_2d = np.concatenate([action[:2], [h]]) + off
        e_2d = np.concatenate([action[2:], [h]]) + off
        lift = np.array([0., 0., self.global_scale / 24.0])
        return [s_2d + lift, s_2d, e_2d, e_2d + lift]

    @staticmethod
    def _pusher_orn(action):
        if (action[0] - action[2]) == 0:
            pusher_angle = np.pi / 2
        else:
            pusher_angle = np.arctan((action[1] - action[3]) / (action[0] - action[2]))
        return np.array([0.0, np.pi, pusher_angle + np.pi / 2])

    def _trajectory(self, action, env_idx):
        """Flatten one env's waypoints into a per-sim-step pose sequence."""
        way_pts = self._waypoints(action, env_idx)
        orn = p.getQuaternionFromEuler(self._pusher_orn(action))
        speed = self.action_step_size
        traj = []
        for i_p in range(len(way_pts) - 1):
            s, e = way_pts[i_p], way_pts[i_p + 1]
            steps = int(np.linalg.norm(e - s) / speed) + 1
            for i in range(steps):
                traj.append(s + (e - s) * i / steps)
        return traj, orn

    def step(self, actions, video_recorder=None):
        """Execute one push per env, all at once.

        `actions`: (n_envs, 4) tile-local [start_x, start_y, end_x, end_y].

        Returns a list of length n_envs: the post-push observation per env, or
        None for an env whose particles went NaN.  Envs run to a common step
        count (shorter pushes hold their final pose), so every env sees the same
        number of solver steps -- that is what makes one shared `pyflex.step()`
        possible.
        """
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (self.n_envs, 4):
            raise ValueError('actions must have shape (%d, 4), got %s'
                             % (self.n_envs, actions.shape))

        # Match FlexEnv.step(): park every arm at the rest pose before planning.
        # Skipping this leaves each robot wherever the previous action ended, and
        # the first pose of the new push then teleports it across the workspace.
        # The teleport is applied twice on purpose: shape states carry a
        # previous-pose slot for continuous collision, so a single call would
        # sweep the arm through the pile between the old pose and the rest pose
        # and detonate the solver (observed: a benign repeated push exploding on
        # its second run).
        self.reset_pandas([self.rest_joints] * self.n_envs)
        self.reset_pandas([self.rest_joints] * self.n_envs)

        trajs, orns = [], []
        for env_idx in range(self.n_envs):
            traj, orn = self._trajectory(actions[env_idx], env_idx)
            trajs.append(traj)
            orns.append(orn)
        max_len = max(len(t) for t in trajs)

        exploded_envs = set()
        exploded = False
        for t in range(max_len):
            joint_positions = []
            for env_idx in range(self.n_envs):
                traj = trajs[env_idx]
                # Envs with shorter pushes hold their last pose rather than
                # retracting early, so the arm stays out of its own pile while
                # the batch finishes.
                ee_pos = traj[min(t, len(traj) - 1)]
                joint_positions.append(p.calculateInverseKinematics(
                    self.robotIds[env_idx],
                    self.end_idx,
                    ee_pos,
                    orns[env_idx],
                    self.joints_lower.tolist(),
                    self.joints_upper.tolist(),
                    (self.joints_upper - self.joints_lower).tolist(),
                    self.rest_joints))
            self.reset_pandas(joint_positions)
            if video_recorder is not None:
                self._record(video_recorder)
            self._sim_step()
            # Check every step by default. The buffer copy is cheap (~0.3ms,
            # about 9% of a clean batch), and bailing out late is expensive:
            # once the solver blows up the particle positions degenerate and
            # each further step costs ~0.8s instead of ~3ms. Measured on a
            # batch containing one explosion: 3.9s at every-25-steps versus
            # 1.0s at every step, same validity outcome. In the 20x120 run,
            # exploding batches were 13% of batches but 67% of total runtime.
            if (t + 1) % self.nan_check_every == 0:
                pos = self.get_positions().reshape(-1, 4)
                if np.isnan(pos[:, 0]).any():
                    exploded_envs = {i for i in range(self.n_envs)
                                     if np.isnan(pos[self._env_slice(i)][:, :3]).any()}
                    print('simulator exploded during push in env(s) %s at step %d/%d'
                          % (sorted(exploded_envs), t + 1, max_len))
                    exploded = True
                    break

        if not exploded:
            # Same teleport concern as above, now on the retract to the neutral
            # pose before settling.
            self.reset_pandas([np.zeros(9).tolist()] * self.n_envs)
            self.reset_pandas([np.zeros(9).tolist()] * self.n_envs)
            for t in range(self.settle_steps):
                if video_recorder is not None:
                    self._record(video_recorder)
                self._sim_step()
                # The settle phase needs the same early bail-out as the push:
                # a wide pusher clipping a wall can destabilise the solver only
                # after it is released, and every step after that costs ~0.15s
                # instead of ~3ms. Without this check the full settle ran to
                # completion on degenerate positions -- measured 30s batches.
                if (t + 1) % self.nan_check_every == 0:
                    pos = self.get_positions().reshape(-1, 4)
                    if np.isnan(pos[:, 0]).any():
                        exploded_envs = {i for i in range(self.n_envs)
                                         if np.isnan(pos[self._env_slice(i)][:, :3]).any()}
                        print('simulator exploded during settle in env(s) %s at step %d/%d'
                              % (sorted(exploded_envs), t + 1, self.settle_steps))
                        exploded = True
                        break

        if exploded:
            # The batch stopped early, so the envs that did NOT explode are
            # mid-push and never settled -- their state is not the outcome of
            # their action and must not be recorded. Worse, stepping again to
            # render (below) with a degenerate tile still in the buffer can
            # push NaN into the healthy tiles, which silently wrote NaN
            # particles into trials marked valid. So: report which envs blew
            # up, return nothing, and let the caller retry the survivors in a
            # later batch (their actions are unaffected by who they batch with).
            self.last_exploded_envs = sorted(exploded_envs)
            self.last_aborted = True
            return [None] * self.n_envs

        self.last_exploded_envs = []
        self.last_aborted = False

        all_pos = self.get_positions().reshape(-1, 4)
        valid = [not bool(np.isnan(all_pos[self._env_slice(i)][:, :3]).any())
                 for i in range(self.n_envs)]
        if not any(valid):
            return [None] * self.n_envs

        if self.render_step_before_capture:
            self._sim_step()
        obs = []
        for env_idx in range(self.n_envs):
            obs.append(self.render_env(env_idx) if valid[env_idx] else None)
        pyflex.set_camPos(self.camPos)
        pyflex.set_camAngle(self.camAngle)
        return obs

    def _record(self, video_recorder):
        """Write the full-grid view to a recorder (debug/inspection only)."""
        pyflex.set_camPos(self.grid_camPos())
        pyflex.set_camAngle(self.camAngle)
        out = self._process_render_output(pyflex.render(render_depth=True))
        video_recorder[0].write(np.ascontiguousarray(
            np.clip(out[..., :3][..., ::-1], 0, 255).astype(np.uint8)))

    def grid_camPos(self):
        """Camera high enough to frame the whole grid (debug views only)."""
        # Grid span edge-to-edge: centers span (n-1)*pitch, plus half a
        # container on each side. The single-env camera sits at 6/8 of the
        # container width; the 1.3 factor is empirical headroom so the outer
        # walls are not clipped.
        span = max((self.grid_nx - 1), (self.grid_nz - 1)) * self.tile_pitch + self.global_scale
        return np.array([0.0, 1.3 * 6.0 * span / 8.0, 0.0])

    def render_grid(self):
        """Single image of the whole grid -- for eyeballing the layout."""
        if self.render_step_before_capture:
            self._sim_step()
        pyflex.set_camPos(self.grid_camPos())
        pyflex.set_camAngle(self.camAngle)
        out = self._process_render_output(pyflex.render(render_depth=True))
        pyflex.set_camPos(self.camPos)
        pyflex.set_camAngle(self.camAngle)
        return out
