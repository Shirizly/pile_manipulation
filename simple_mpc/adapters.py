"""
Model adapters for run_simple_mpc.

Each adapter wraps one model type and exposes five model-agnostic operations:

  obs_to_state       : rendered obs (H,W,5)  →  state tensor on device
                       Also updates any per-step bookkeeping (e.g. particle_dens)
  expand_state       : (1,*S)                →  (n_sample,*S) clone, grad-ready
  predict_step       : (n_sample,*S), (n_sample,4) → (n_sample,*S)  [backprop-able]
  compute_reward     : (n_sample,*S)         →  (n_sample,)          [backprop-able]
  format_states_pred : list of per-step state ndarrays → stored in output dict

Optionally:
  print_step_info    : prints model-specific material / coverage diagnostics
  debug_vis_enabled  : bool — whether the run_simple_mpc debug_vis hooks fire

The factory function ``make_adapter`` auto-selects the right adapter for the
model type.  To support a new model, subclass one of these or write your own
class with the same API and register it in ``make_adapter``.
"""

from __future__ import annotations

import numpy as np
import math
import torch

from utils import depth2fgpcd, fps_np
from transforms.functional import particles_to_occupancy as _particles_to_occupancy
from simple_mpc.occupancy_reward import OccupancyReward

# Depth threshold for foreground detection; must match utils.scale_subgoal_to_material_pixels
_FG_DEPTH_THRESHOLD = 0.599 / 0.8


# ─────────────────────────────────────────────────────── geometry helper ─────

def _gen_s_delta(s_cur: torch.Tensor,
                 action: torch.Tensor,
                 cam_extrinsic: np.ndarray,
                 global_scale: float,
                 pusher_w: float) -> torch.Tensor:
    """
    Compute per-particle displacement field for a world-space push action.

    Extracted from ``PlannerGD.gen_s_delta``; kept standalone to avoid importing
    the planner (which has heavy env dependencies).

    Parameters
    ----------
    s_cur        : (N, particle_num, 3) camera-space particle positions
    action       : (N, 4)  [sx, sy, ex, ey] in world 2-D coords
    cam_extrinsic: (4, 4)  world-to-camera affine transform (from env.get_cam_extrinsics())
    global_scale : float   normalisation factor for camera-space coords
    pusher_w     : float   half-width of the pusher (in normalised camera units)

    Returns
    -------
    (N, particle_num, 3) per-particle displacement to add to s_cur
    """
    N     = action.shape[0]
    dev   = action.device
    dtype = action.dtype

    s_xy = action[:, :2]   # (N, 2)
    e_xy = action[:, 2:]   # (N, 2)
    h    = torch.zeros((N, 1), device=dev, dtype=dtype)  # floor height

    # World 3-D: x_world, y_up=0, z_world = -y_2d  (matches PlannerGD convention)
    s_3d = torch.cat([s_xy[:, 0:1],  h, -s_xy[:, 1:2]], dim=1)   # (N, 3)
    e_3d = torch.cat([e_xy[:, 0:1],  h, -e_xy[:, 1:2]], dim=1)   # (N, 3)

    # World → camera-space transform (equivalent to PlannerGD.world2cam):
    #   cam_coords = (inv(opencv_T_opengl) @ cam_extrinsic @ [world | 1].T).T / global_scale
    # which simplifies to   M = opencv_T_opengl.T @ cam_extrinsic  (since B^{-1}=B for axis-flip)
    opencv_T_opengl = np.array([[1, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]], dtype=np.float64)
    M = (opencv_T_opengl.T @ cam_extrinsic).astype(np.float32)   # (4, 4)
    M_t = torch.tensor(M, device=dev, dtype=dtype)

    def _w2c(pts3d):   # pts3d: (N, 3)
        ones = torch.ones((pts3d.shape[0], 1), device=dev, dtype=dtype)
        return (M_t @ torch.cat([pts3d, ones], dim=1).T).T[:, :3] / global_scale

    s_cam = _w2c(s_3d)   # (N, 3) — camera-space start
    e_cam = _w2c(e_3d)   # (N, 3) — camera-space end

    push_vec = e_cam - s_cam                                        # (N, 3)
    push_l   = torch.linalg.norm(push_vec, dim=1, keepdim=True)    # (N, 1)
    push_dir = push_vec / push_l                                    # (N, 3)

    # Lateral orthogonal direction (XY plane only)
    zeros  = torch.zeros((N, 1), device=dev, dtype=dtype)
    ortho  = torch.cat([-push_dir[:, 1:2], push_dir[:, 0:1], zeros], dim=1)  # (N, 3)

    # Per-particle projections onto longitudinal and lateral axes
    diff       = s_cur - s_cam[:, None, :]                          # (N, particle_num, 3)
    proj_l     = (diff * push_dir[:, None, :]).sum(-1)              # (N, particle_num)
    proj_ortho = (diff *      ortho[:, None, :]).sum(-1)            # (N, particle_num)

    # Longitudinal gate: particle must be in [0, push_l] along the push direction
    l_mask = ((proj_l > 0.0) & (proj_l < push_l[:, 0:1])).float()  # hard   (N, particle_num)

    # Lateral soft gate: Gaussian falloff outside ±pusher_w
    excess = torch.maximum(torch.clamp(-pusher_w - proj_ortho, min=0.),
                           torch.clamp( proj_ortho - pusher_w, min=0.))
    w_mask = torch.exp(-excess / 0.01)                              # (N, particle_num)

    # Displacement along push direction: how far does the particle have to move to the end?
    to_end      = e_cam[:, None, :] - s_cur                         # (N, particle_num, 3)
    proj_to_end = (to_end * push_dir[:, None, :]).sum(-1)           # (N, particle_num)

    s_delta = (proj_to_end[..., None]                               # (N, particle_num, 3)
               * push_dir[:, None, :]
               * l_mask[..., None]
               * w_mask[..., None])
    return s_delta


# ─────────────────────────────────────── Eulerian adapter ─────────────────────

class EulerianAdapter:
    """Wraps EulerianModelWrapper for run_simple_mpc."""

    debug_vis_enabled = True   # full debug_vis pipeline is available

    # Reward type registry: name → (callable, valid_for_opt, valid_for_report)
    # valid_for_opt/report indicate which contexts the reward is suitable for
    _REWARD_METHODS = {
        'default': ('_reward_default', True, True),
        'iou':     ('_reward_iou', True, True),
    }

    def __init__(self,
                 model_dy,
                 subgoal: np.ndarray,
                 cam_params,
                 global_scale: float,
                 device: str,
                 empty_penalty: float = 0.0,
                 reward_type_opt: str = 'default',
                 reward_type_report: str | None = None):
        self.model_dy     = model_dy
        self.cam_params   = cam_params
        self.global_scale = global_scale
        self.device       = device if torch.cuda.is_available() else 'cpu'

        # Pre-compute goal score map (fixed throughout the run).
        self.score_tensor = model_dy.prepare_goal_reward(
            subgoal, cam_params, device=device, empty_penalty=empty_penalty)
        self.score_np     = self.score_tensor.cpu().numpy()    # (Nx, Ny) for debug vis

        # Set reward type configuration
        self.reward_type_opt = reward_type_opt
        self.reward_type_report = reward_type_report or reward_type_opt

        # Validate reward types
        for rt in [self.reward_type_opt, self.reward_type_report]:
            if rt not in self._REWARD_METHODS:
                raise ValueError(
                    f"Unknown Eulerian reward type '{rt}'. "
                    f"Available: {list(self._REWARD_METHODS.keys())}"
                )
            _, valid_opt, valid_report = self._REWARD_METHODS[rt]
            if rt == self.reward_type_opt and not valid_opt:
                raise ValueError(
                    f"Reward type '{rt}' is not valid for optimization"
                )
            if rt == self.reward_type_report and not valid_report:
                raise ValueError(
                    f"Reward type '{rt}' is not valid for reporting"
                )

    # ── public API ────────────────────────────────────────────────────────────

    def _get_example_state_shape(self) -> tuple:
        """Return state shape (without batch dimension) for benchmarking."""
        return (self.model_dy.grid_res[0], self.model_dy.grid_res[1])

    def obs_to_state(self, obs_np: np.ndarray) -> torch.Tensor:
        """(H,W,5) → (1, Nx, Ny) occupancy tensor (detached, on device)."""
        depth = obs_np[..., -1] / self.global_scale
        pts   = depth2fgpcd(depth, depth < _FG_DEPTH_THRESHOLD, self.cam_params)
        pts_t = torch.from_numpy(pts).float().to(self.device).unsqueeze(0)
        with torch.no_grad():
            occ = self.model_dy.initial_occ_from_particles(pts_t)
        return occ.detach()                                    # (1, Nx, Ny)

    def expand_state(self, state: torch.Tensor, n_sample: int) -> torch.Tensor:
        """(1, Nx, Ny) → (n_sample, Nx, Ny) — cloned so gradients can flow."""
        return state.expand(n_sample, *state.shape[1:]).clone()

    def predict_step(self, state_batch: torch.Tensor,
                     act_batch: torch.Tensor) -> torch.Tensor:
        """(n_sample, Nx, Ny) × (n_sample, 4) → (n_sample, Nx, Ny)."""
        return self.model_dy.predict_one_step_occ(state_batch, act_batch)

    def compute_reward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """(n_sample, Nx, Ny) → (n_sample,) reward (higher = closer to goal)."""
        return (state_batch.clamp(0.0, 1.0) * self.score_tensor
                ).reshape(state_batch.shape[0], -1).sum(dim=-1)

    def compute_reward_iou(self, state_batch: torch.Tensor) -> torch.Tensor:
        """(n_sample, Nx, Ny) → (n_sample,) IoU reward with goal region.
        
        Computes intersection over union between predicted material
        (occupancy > 0.5) and goal region (score_tensor > 0).
        Works with score tensors marked as 0 or -1 for empty areas.
        """
        # Binarize: predicted material where occupancy > 0.5
        pred_material = (state_batch > 0.5).float()  # (n_sample, Nx, Ny)
        
        # Goal region where score tensor is positive (1=goal, 0 or -1=empty)
        goal_material = (self.score_tensor > 0).float()  # (Nx, Ny)
        
        # Flatten for batch computation
        pred_flat = pred_material.view(pred_material.shape[0], -1)  # (n_sample, Nx*Ny)
        goal_flat = goal_material.view(-1)  # (Nx*Ny,)
        
        # Intersection and union
        intersection = (pred_flat * goal_flat).sum(dim=1)  # (n_sample,)
        union = torch.maximum(pred_flat, goal_flat.unsqueeze(0)).sum(dim=1)  # (n_sample,)
        
        # IoU (add epsilon to avoid division by zero)
        iou = intersection / (union + 1e-6)
        return iou

    # ── private reward implementations ─────────────────────────────────────

    def _reward_default(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Default reward: weighted sum of occupancy × goal score map."""
        return (state_batch.clamp(0.0, 1.0) * self.score_tensor
                ).reshape(state_batch.shape[0], -1).sum(dim=-1)

    def _reward_iou(self, state_batch: torch.Tensor) -> torch.Tensor:
        """IoU reward (delegates to compute_reward_iou)."""
        return self.compute_reward_iou(state_batch)

    # ── reward getter interface ────────────────────────────────────────────

    def obs_to_report_state(self, obs_np: np.ndarray) -> torch.Tensor:
        """Return state for reporting reward evaluation.

        For EulerianAdapter the reporting state is the same as the MPC state:
        both are derived from the full depth image via initial_occ_from_particles,
        so density is already maximal.
        """
        return self.obs_to_state(obs_np)

    def get_reward_fn_opt(self) -> callable:
        """Return the reward function to use for optimization."""
        method_name = self._REWARD_METHODS[self.reward_type_opt][0]
        return getattr(self, method_name)

    def get_reward_fn_report(self) -> callable:
        """Return the reward function to use for reporting."""
        method_name = self._REWARD_METHODS[self.reward_type_report][0]
        return getattr(self, method_name)

    def format_states_pred(self, rollout_seq: list) -> np.ndarray:
        """list[(Nx,Ny) ndarray] × n_ahead → (n_ahead, Nx, Ny) for output dict."""
        return np.stack(rollout_seq)

    def print_step_info(self, state: torch.Tensor, step: int, n_mpc: int):
        """Print material-coverage diagnostics using the occupancy grid."""
        occ_np = state[0].cpu().numpy()
        mask   = occ_np > 0.5
        cov    = mask.mean() * 100
        if mask.any():
            ci, cj = np.where(mask)
            print(f"  step {step}: material {cov:.1f}% of grid, "
                  f"centroid voxel ({ci.mean():.1f}, {cj.mean():.1f}) "
                  f"of {occ_np.shape}  "
                  f"[action range: x_grid 0-{occ_np.shape[0]-1}, "
                  f"y_grid 0-{occ_np.shape[1]-1}]")
        else:
            print(f"  step {step}: WARNING — occupancy grid is entirely empty! "
                  "Check depth threshold and grid bounds.")

    def debug_occ(self, state: torch.Tensor) -> np.ndarray:
        """Return occupancy as numpy (Nx, Ny) for debug_vis helpers."""
        return state[0].cpu().numpy()

    def get_ot_grids(self, state: torch.Tensor) -> tuple:
        """Return (source_grid, goal_grid) as (H, W) float32 numpy arrays.

        ``source_grid`` is the current occupancy; ``goal_grid`` is a binary
        mask of the goal region (1 inside goal, 0 elsewhere).
        """
        source_grid = state.squeeze(0).cpu().numpy().astype(np.float32)
        goal_grid   = (self.score_np > 0).astype(np.float32)
        return source_grid, goal_grid


# ─────────────────────────────────────────── GNN adapter ─────────────────────

class GNNAdapter:
    """
    Wraps PropNetDiffDenModel for run_simple_mpc.

    The GNN model operates on fixed-size particle point clouds (particle_num × 3).
    Action encoding follows PlannerGD.gen_s_delta: each 4-D world-space action
    [sx, sy, ex, ey] is converted into a per-particle displacement field
    (the 'selta' used as GNN input), which is differentiable w.r.t. the action.

    For MPC optimization, a particle-based reward matching the state
    representation is intended, but the FlexEnv implementation
    (config_reward_ptcl) has not been ported to Genesis yet — the 'default'
    optimization reward raises NotImplementedError (see INTERFACES.md §6).

    For reporting/comparison, computes occupancy-based reward from the full raw
    observation (compute_occ_reward_from_obs) so it is on the same scale as the
    Eulerian adapter for model-invariant comparison.

    Required config keys
    --------------------
    cfg['mpc']['particle_num']        (default 50)
    cfg['dataset']['global_scale']
    """

    debug_vis_enabled = False   # debug_vis uses occupancy arrays — skip for GNN

    # Reward type registry: name → (callable, valid_for_opt, valid_for_report)
    # GNN uses particle reward for optimization, can use either for reporting
    _REWARD_METHODS = {
        'default':   ('_reward_default', True,  True),
        'iou':       ('_reward_iou',     False, True),  # report only: needs occ grid
        'eulerian':  ('_reward_eulerian', False, True),  # report only: particles → occ × score
    }

    def __init__(self,
                 model_dy,
                 env,
                 subgoal: np.ndarray,
                 cam_params,
                 cfg: dict,
                 device: str = 'cuda',
                 reward_type_opt: str = 'default',
                 reward_type_report: str | None = None):
        self.model_dy      = model_dy
        self.env           = env
        self.cam_params    = cam_params
        self.global_scale  = cfg['dataset']['global_scale']
        self.device        = device
        self.particle_num  = cfg['mpc'].get('particle_num', 50)
        self.pusher_w      = 0.8 / self.global_scale   # matches PlannerGD convention
        self.cam_extrinsic = env.get_cam_extrinsics()  # (4, 4) ndarray — TBD: GenesisEnv returns np.eye(4); genesis action convention does not use extrinsic matrix
        # The FlexEnv particle reward (config_reward_ptcl) was never ported to
        # Genesis, so the 'default' optimization reward is unavailable; see
        # INTERFACES.md §6.  _reward_default fails loudly instead of silently.
        self._reward_fn    = None

        # particle density — updated each obs_to_state call
        self._particle_dens: float = 1.0

        # Pre-compute goal tensors (fixed throughout the run).
        subgoal_t = torch.from_numpy(subgoal).float().to(device)
        H, W      = subgoal_t.shape

        # goal_coor: (col, row) pixel positions of goal-interior pixels
        goal_coor_np = torch.flip((subgoal_t < 0.5).nonzero(),
                                  dims=(1,)).float().cpu().numpy()
        n_goal = min(self.particle_num * 5, goal_coor_np.shape[0])
        if n_goal > 0 and goal_coor_np.shape[0] > n_goal:
            goal_coor_np, _ = fps_np(goal_coor_np, n_goal)

        self.goal_t      = subgoal_t
        self.goal_coor_t = torch.from_numpy(goal_coor_np).float().to(device)

        # Set reward type configuration
        self.reward_type_opt = reward_type_opt
        self.reward_type_report = reward_type_report or reward_type_opt

        # Validate reward types
        for rt in [self.reward_type_opt, self.reward_type_report]:
            if rt not in self._REWARD_METHODS:
                raise ValueError(
                    f"Unknown GNN reward type '{rt}'. "
                    f"Available: {list(self._REWARD_METHODS.keys())}"
                )
            _, valid_opt, valid_report = self._REWARD_METHODS[rt]
            if rt == self.reward_type_opt and not valid_opt:
                raise ValueError(
                    f"Reward type '{rt}' is not valid for optimization in GNN adapter. "
                    f"Use 'default' for optimization."
                )
            if rt == self.reward_type_report and not valid_report:
                raise ValueError(
                    f"Reward type '{rt}' is not valid for reporting in GNN adapter"
                )

        # Store occupancy grid parameters for compute_occ_reward_from_obs()
        self.grid_bounds = None
        self.grid_res = None
        self.occ_score_tensor = None

    # ── public API ────────────────────────────────────────────────────────────

    def _get_example_state_shape(self) -> tuple:
        """Return state shape (without batch dimension) for benchmarking."""
        return (self.particle_num, 3)

    def obs_to_state(self, obs_np: np.ndarray) -> torch.Tensor:
        """
        (H,W,5) → (1, particle_num, 3) particle tensor (detached, on device).

        Uses a small internal batch to get a robust particle_r estimate, then
        updates self._particle_dens for subsequent predict_step calls.
        """
        # TBD: obs2ptcl_fixed_num_batch is FlexEnv-specific — needs Genesis equivalent
        pts_batch, r_batch = self.env.obs2ptcl_fixed_num_batch(
            obs_np, self.particle_num, batch_size=5)
        particle_r          = float(np.median(r_batch))
        self._particle_dens = 1.0 / (particle_r ** 2)

        pts0 = pts_batch[0]    # (particle_num, 3) — first sampled version
        return (torch.from_numpy(pts0).float()
                .to(self.device).unsqueeze(0).detach())   # (1, particle_num, 3)

    def expand_state(self, state: torch.Tensor, n_sample: int) -> torch.Tensor:
        """(1, particle_num, 3) → (n_sample, particle_num, 3)."""
        return state.expand(n_sample, *state.shape[1:]).clone()

    def predict_step(self, state_batch: torch.Tensor,
                     act_batch: torch.Tensor) -> torch.Tensor:
        """
        (n_sample, particle_num, 3) × (n_sample, 4) → (n_sample, particle_num, 3).

        Gradient flows through both _gen_s_delta (geometry) and
        GNN.predict_one_step (learned dynamics).
        """
        n     = state_batch.shape[0]
        a_cur = torch.zeros(n, self.particle_num,
                            device=self.device, dtype=state_batch.dtype)
        dens  = torch.full((n,), self._particle_dens,
                           device=self.device, dtype=state_batch.dtype)
        s_delta = _gen_s_delta(state_batch, act_batch,
                               self.cam_extrinsic, self.global_scale,
                               self.pusher_w)
        return self.model_dy.predict_one_step(a_cur, state_batch, s_delta, dens)

    def compute_reward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """(n_sample, particle_num, 3) → (n_sample,) reward.

        Uses particle-based reward directly, preserving gradients through the
        optimization. This matches the state representation (particles).
        """
        return self._reward_default(state_batch)

    def compute_reward_iou(self, state_batch: torch.Tensor) -> torch.Tensor:
        """(n_sample, particle_num, 3) → (n_sample,) IoU reward with goal region.
        
        Converts particles to occupancy grid, then computes IoU between
        predicted material (occupancy > 0.5) and goal region (occ_score_tensor > 0).
        Works with score tensors marked as 0 or -1 for empty areas.
        """
        if self.grid_bounds is None or self.occ_score_tensor is None:
            raise RuntimeError(
                "IoU reward requires occupancy parameters to be set via "
                "set_occupancy_params(). This should be called automatically "
                "by make_adapter(); check that occupancy initialization succeeded."
            )
        
        # Convert particles to occupancy
        occ = _particles_to_occupancy(
            state_batch, self.grid_bounds, self.grid_res, sigma=0.0)
        # (n_sample, Nx, Ny)
        
        # Binarize: predicted material where occupancy > 0.5
        pred_material = (occ > 0.5).float()  # (n_sample, Nx, Ny)
        
        # Goal region where score tensor is positive (1=goal, 0 or -1=empty)
        goal_material = (self.occ_score_tensor > 0).float()  # (Nx, Ny)
        
        # Flatten for batch computation
        pred_flat = pred_material.view(pred_material.shape[0], -1)  # (n_sample, Nx*Ny)
        goal_flat = goal_material.view(-1)  # (Nx*Ny,)
        
        # Intersection and union
        intersection = (pred_flat * goal_flat).sum(dim=1)  # (n_sample,)
        union = torch.maximum(pred_flat, goal_flat.unsqueeze(0)).sum(dim=1)  # (n_sample,)
        
        # IoU (add epsilon to avoid division by zero)
        iou = intersection / (union + 1e-6)
        return iou

    # ── private reward implementations ─────────────────────────────────────

    def _reward_default(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Default particle-based reward (delegates to the ported reward fn)."""
        if self._reward_fn is None:
            raise NotImplementedError(
                "GNNAdapter's default particle reward is not available: the "
                "FlexEnv config_reward_ptcl function has not been ported to "
                "Genesis (see INTERFACES.md §6). Use reward_type_report="
                "'iou'/'eulerian' for reporting, or port a particle reward and "
                "assign it to self._reward_fn."
            )
        return self._reward_fn(
            state_batch,
            self.goal_t,
            cam_params=self.cam_params,
            goal_coor=self.goal_coor_t,
            normalize=True,
        )

    def _reward_iou(self, state_batch: torch.Tensor) -> torch.Tensor:
        """IoU reward via occupancy conversion (delegates to compute_reward_iou)."""
        return self.compute_reward_iou(state_batch)

    def _reward_eulerian(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Eulerian-style reward: convert particles → occupancy, then apply occ_score_tensor.

        Produces rewards on the same scale as EulerianAdapter._reward_default,
        enabling direct cross-model comparison.
        Requires occupancy parameters to be set via set_occupancy_params().
        """
        if self.grid_bounds is None or self.occ_score_tensor is None:
            raise RuntimeError(
                "'eulerian' reward type requires occupancy parameters to be set via "
                "set_occupancy_params(). This should be called automatically "
                "by make_adapter(); check that occupancy initialization succeeded."
            )
        occ = _particles_to_occupancy(
            state_batch, self.grid_bounds, self.grid_res, sigma=0.0)  # (n_sample, Nx, Ny)
        return (occ.clamp(0.0, 1.0) * self.occ_score_tensor
                ).reshape(occ.shape[0], -1).sum(dim=-1)  # (n_sample,)

    # ── reward getter interface ────────────────────────────────────────────

    def get_reward_fn_opt(self) -> callable:
        """Return the reward function to use for optimization."""
        method_name = self._REWARD_METHODS[self.reward_type_opt][0]
        return getattr(self, method_name)

    def get_reward_fn_report(self) -> callable:
        """Return the reward function to use for reporting."""
        method_name = self._REWARD_METHODS[self.reward_type_report][0]
        return getattr(self, method_name)

    def compute_occ_reward_from_obs(self, obs_np: np.ndarray) -> float | None:
        """
        Compute occupancy-based reward from a raw rendered observation.

        For reporting/comparison only (not used during MPC optimization).
        Uses the FULL foreground point cloud extracted from the depth channel,
        matching the density of what the Eulerian adapter sees, so rewards are
        on the same scale across model types.

        Returns None if occupancy parameters were not set.
        """
        if self.grid_bounds is None or self.occ_score_tensor is None:
            return None

        depth = obs_np[..., -1] / self.global_scale
        pts_np = depth2fgpcd(depth, depth < _FG_DEPTH_THRESHOLD, self.cam_params)
        if pts_np.shape[0] == 0:
            return 0.0

        pts_t = torch.from_numpy(pts_np.astype(np.float32)).to(self.device).unsqueeze(0)
        # Use hard-voxel splatting (sigma=0) — same as EulerianAdapter.obs_to_state
        occ = _particles_to_occupancy(pts_t, self.grid_bounds, self.grid_res, sigma=0.0)
        return float((occ.clamp(0.0, 1.0) * self.occ_score_tensor).reshape(1, -1).sum().item())

    def set_occupancy_params(self, grid_bounds: dict, grid_res: tuple,
                            occ_score_tensor: torch.Tensor):
        """
        Set occupancy grid parameters for compute_occ_reward_from_obs().

        Called by make_adapter to enable observation-based occupancy reporting.
        """
        self.grid_bounds = grid_bounds
        self.grid_res = grid_res
        self.occ_score_tensor = occ_score_tensor

    def format_states_pred(self, rollout_seq: list) -> np.ndarray:
        """list[(particle_num,3) ndarray] × n_ahead → (particle_num,3) last step."""
        return rollout_seq[-1]   # matches step_subgoal_ptcl's states_pred format

    def obs_to_report_state(self, obs_np: np.ndarray) -> torch.Tensor:
        """Return state for reporting reward evaluation.

        For eulerian and iou report types the reward operates on an occupancy
        grid derived from particles, so the quality of the measurement scales
        directly with the number of input points.  Using the coarse 50-particle
        MPC state yields a severely undersampled occupancy grid (<1.2 % voxel
        fill) with high per-step variance that obscures real progress.

        This method returns the *full* foreground point cloud extracted from the
        depth channel (same source used by EulerianAdapter.obs_to_state), so
        that _reward_eulerian / _reward_iou produce dense, low-noise measurements
        on the same scale as EulerianAdapter._reward_default.

        For the default report type the MPC particle state is sufficient, so
        this delegates to obs_to_state.
        """
        if self.reward_type_report in ('eulerian', 'iou'):
            depth  = obs_np[..., -1] / self.global_scale
            pts_np = depth2fgpcd(depth, depth < _FG_DEPTH_THRESHOLD,
                                 self.cam_params)
            if pts_np.shape[0] == 0:
                return torch.zeros((1, 1, 3), device=self.device)
            return (torch.from_numpy(pts_np.astype(np.float32))
                    .to(self.device).unsqueeze(0).detach())  # (1, N_full, 3)
        return self.obs_to_state(obs_np)

    def print_step_info(self, state: torch.Tensor, step: int, n_mpc: int):
        """Print particle-count diagnostics."""
        n_particles = state.shape[1]
        print(f"  step {step}: {n_particles} particles  "
              f"particle_dens={self._particle_dens:.2f}")

    def get_ot_grids(self, state: torch.Tensor) -> tuple:
        """OT grids not available for GNN adapter; returns (None, None)."""
        return None, None


# ─────────────────────────────────────────────────── factory ─────────────────

def make_adapter(model_dy,
                 env,
                 subgoal: np.ndarray,
                 cfg: dict,
                 cam_params,
                 device: str = 'cuda'):
    """
    Return the appropriate ModelAdapter for *model_dy*.

    Each adapter uses the reward function that naturally matches its state
    representation:
    - Eulerian: occupancy-based reward (state is occupancy grid)
    - GNN: particle-based reward (state is particles)

    Occupancy-based metrics can be computed separately for reporting/comparison.

    Parameters
    ----------
    model_dy   : the dynamics model (EulerianModelWrapper or PropNetDiffDenModel)
    env        : FlexEnv (already initialised and reset)
    subgoal    : (H, W) float32 distance-transform;  0 = inside goal
    cfg        : full config dict
    cam_params : camera intrinsics tuple from env.get_cam_params()
    device     : 'cuda' or 'cpu'
    """
    from model.eulerian_wrapper import EulerianModelWrapper
    from model.gnn_dyn import PropNetDiffDenModel

    reward_cfg      = cfg.get('mpc', {}).get('reward', {})
    empty_penalty   = float(reward_cfg.get('empty_penalty', 0.0))
    reward_type_opt = str(reward_cfg.get('opt_type', 'default'))
    reward_type_rep = reward_cfg.get('report_type', None)
    if reward_type_rep is not None:
        reward_type_rep = str(reward_type_rep)
    global_scale    = cfg['dataset']['global_scale']

    if isinstance(model_dy, EulerianModelWrapper):
        return EulerianAdapter(model_dy, subgoal, cam_params,
                               global_scale, device,
                               empty_penalty=empty_penalty,
                               reward_type_opt=reward_type_opt,
                               reward_type_report=reward_type_rep)

    if isinstance(model_dy, PropNetDiffDenModel):
        # GNN: create adapter with particle-based reward for optimization
        adapter = GNNAdapter(model_dy, env, subgoal, cam_params, cfg, device,
                             reward_type_opt=reward_type_opt,
                             reward_type_report=reward_type_rep)

        # Optionally set occupancy parameters for reporting (if needed)
        grid_bounds = EulerianModelWrapper.default_bounds(cfg)
        grid_res = (64, 64)  # standard occupancy grid resolution
        occ_reward = OccupancyReward(grid_bounds, grid_res, global_scale, cam_params)
        occ_score_tensor = occ_reward.compute_score_tensor(
            subgoal, device=device, empty_penalty=empty_penalty)
        adapter.set_occupancy_params(grid_bounds, grid_res, occ_score_tensor)

        return adapter

    raise NotImplementedError(
        f"No simple_mpc adapter for model type '{type(model_dy).__name__}'. "
        "Implement a ModelAdapter and register it in make_adapter(). "
        "For the Baselines/ occupancy-grid models (NFD family, switched-linear "
        "operators), which are not driven from a rendered observation at all, "
        "use make_occ_adapter() / OCC_ADAPTERS further down this module."
    )


# ═════════════════════════════════════════════════════════════════════════════
#  Occupancy-grid GRADIENT adapters  (EXP-0023)
# ═════════════════════════════════════════════════════════════════════════════
#
# WHY A SECOND FAMILY.  The adapters above are built around the live
# `run_simple_mpc` loop: they take a rendered `(H,W,5)` observation, own a
# camera/`global_scale` convention, and score against a distance-transform
# subgoal.  The baseline models (`Baselines/NFD/*`, `Baselines/LinearForesight/*`)
# do not live in that world at all -- they consume a 64x64 world-frame
# occupancy grid built straight from particle positions, over the fixed
# +/-64 mm slate workspace, and they are scored with `control_utility_test
# .lyapunov`.  Their published `predict_occ(batch)` entry points are an
# OFFLINE SCORING interface: they are `@torch.no_grad()` and return CPU
# tensors, so they cannot be optimised against.
#
# The adapters below close exactly that gap.  Contract (mirrors the five
# operations at the top of this module, minus the camera):
#
#   state_from_particles(states)  : (B,n_particles,>=3) world metres -> (B,H,W)
#   predict_step(occ, act)        : (B,H,W) x (B,4) world-metre [sx,sy,ex,ey]
#                                   -> (B,H,W), DIFFERENTIABLE W.R.T. `act`
#   value(occ)                    : (B,H,W) -> (B,)  Lyapunov value
#   dv(occ0, act)                 : value(predict_step) - value(occ0)
#
# SIGN CONVENTION, FIXED ONCE AND ASSERTED (`assert_dv_convention` below):
#   dv = value(after) - value(before).  It is a COST: negative means the push
#   moved material TOWARDS the goal, and LOWER IS BETTER.  This is the same
#   convention as `scripts/probes/binned_pool_cache.py` and every ranking
#   metric in `experiments/METRICS.md`.  An optimiser MINIMISES `dv`.
#
# NO MATH IS DUPLICATED for the NFD family.  Each NFD adapter wraps the
# already-validated predictor object and calls its `predict_occ` through
# `__wrapped__`, i.e. the undecorated function `torch.no_grad()` wrapped --
# so the adapter runs the IDENTICAL forward the offline eval runs, with
# grad-mode simply left on.  If the predictor's math changes, the adapter
# changes with it; they cannot drift.

from dataclasses import dataclass as _dataclass

from transforms.functional import action_to_pose as _action_to_pose

# The slate-corpus occupancy convention, identical to
# `scripts/probes/binned_pool_cache.py` and `experiments/temp/multistep-rollout
# /rollout.py`, so occupancies (and therefore dv) are on the same scale as
# every number already recorded against these models.
OCC_BOUNDS = {"x_min": -0.064, "x_max": 0.064, "y_min": -0.064, "y_max": 0.064}
OCC_GRID = 64
OCC_CUBE_SIZE = 0.005
OCC_PITCH = (OCC_BOUNDS["x_max"] - OCC_BOUNDS["x_min"]) / OCC_GRID
OCC_FOOTPRINT_RADIUS = 0.5 * OCC_CUBE_SIZE / OCC_PITCH


class SlateRawStub:
    """The two lines of dataset geometry `Baselines/NFD/predictor.py`'s
    predictors read off a `PileSweepData` (`to_pxl`, `ctr_in_PXL`,
    `resolution_scale`, `configs[0]['plate']['size']`), for the fixed
    +/-64 mm / 64 px slate workspace -- so a predictor can be driven from raw
    actions without constructing a dataset.  Same values as
    `experiments/temp/multistep-rollout/rollout.py::RawStub`, promoted here
    because it is not experiment-specific."""
    to_pxl = OCC_GRID / (OCC_BOUNDS["x_max"] - OCC_BOUNDS["x_min"])
    ctr_in_PXL = torch.tensor([OCC_GRID / 2, OCC_GRID / 2, 0.0])
    resolution_scale = 0.5
    configs = [{"plate": {"size": [0.04, 0.002, 0.01]}}]


@_dataclass
class _OccBatch:
    """Duck-typed `Baselines.common.eval_baseline.PredictorBatch` carrying a
    GRAD-CARRYING action.  Only the fields the NFD-family predictors read are
    populated (`occ0`, `p_start`, `p_stop`, `angle`, `raw`, `H`, `W`)."""
    occ0: "torch.Tensor"
    p_start: "torch.Tensor"
    p_stop: "torch.Tensor"
    angle: "torch.Tensor"
    raw: object
    H: int
    W: int


def occ_from_particles(states: "torch.Tensor", device=None) -> "torch.Tensor":
    """(B, n_particles, >=3) world-metre particle states -> (B, 64, 64)."""
    pts = states[..., :3]
    if device is not None:
        pts = pts.to(device)
    return _particles_to_occupancy(pts.float(), OCC_BOUNDS, (OCC_GRID, OCC_GRID),
                                   footprint_radius=OCC_FOOTPRINT_RADIUS)


def occ_for_scoring(states: "torch.Tensor", device=None) -> "torch.Tensor":
    """(B, n_particles, >=3) world-metre particle states -> (B, 64, 64) SCORING
    density on the same grid and axis convention as `occ_from_particles`
    (dim 0 = world x, `particles_to_occupancy`'s (x-lo)/(hi-lo)*(res-1)
    pixel-centre mapping), but drawn with the mass-conserving
    `transforms.functional.splat_particles_mass` instead of the hard
    footprint, scaled so one particle carries the hard footprint's mean
    pixel mass. Use it for TRUE outcomes (ground truth read from particle
    states); `occ_from_particles` stays the model-input representation.
    See experiments/METRICS.md, "Ground-truth scoring" (EXP-0027)."""
    from transforms.functional import splat_particles_mass
    pts = states[..., :3].float()
    if device is not None:
        pts = pts.to(device)
    lo = torch.tensor([OCC_BOUNDS["x_min"], OCC_BOUNDS["y_min"]], device=pts.device)
    hi = torch.tensor([OCC_BOUNDS["x_max"], OCC_BOUNDS["y_max"]], device=pts.device)
    uv = (pts[..., :2] - lo) / (hi - lo) * (OCC_GRID - 1)
    return splat_particles_mass(uv, (OCC_GRID, OCC_GRID), mass=math.pi * OCC_FOOTPRINT_RADIUS ** 2)


class OccupancyGradientAdapter:
    """Base class: everything except `predict_step`.

    `goal_shape` selects the Lyapunov weight field
    (`control_utility_test.lyapunov_weights`)."""

    def __init__(self, name: str, device: str = "cuda", goal_shape: str = "corner"):
        from control_utility_test import lyapunov_weights
        self.name = name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.goal_shape = goal_shape
        self.dw = lyapunov_weights((OCC_GRID, OCC_GRID), goal_shape, self.device)
        self.raw = SlateRawStub()
        # The value function this adapter scores with, and its sense. A
        # subclass that swaps `value` for a mass function must change BOTH;
        # `assert_dv_convention` checks the pair against goal geometry.
        self.value_fn = "lyapunov"
        self.higher_is_better = False

    # ── state ────────────────────────────────────────────────────────────────
    def state_from_particles(self, states):
        return occ_from_particles(states, self.device)

    def expand_state(self, state, n_sample):
        return state.expand(n_sample, *state.shape[1:]).clone()

    # ── value ────────────────────────────────────────────────────────────────
    def value(self, occ):
        """(B,H,W) -> (B,) Lyapunov value. LOWER IS BETTER."""
        from control_utility_test import lyapunov
        return lyapunov(occ, self.dw)

    def dv(self, occ0, act):
        """dv = value(after) - value(before). A COST: lower is better.
        Differentiable w.r.t. `act`."""
        occ1 = self.predict_step(occ0, act)
        return self.value(occ1) - self.value(occ0)

    # ── model ────────────────────────────────────────────────────────────────
    def predict_step(self, occ, act):
        raise NotImplementedError

    # ── helpers ──────────────────────────────────────────────────────────────
    def _batch(self, occ, act):
        """(B,H,W) occupancy + (B,4) world-metre action -> `_OccBatch`, with
        the plate yaw derived from the travel direction exactly as
        `transforms.functional.action_to_pose` (and the corpus's own
        `angles`) define it -- differentiably."""
        sx, sy, ex, ey, angle = _action_to_pose(act)
        z = torch.zeros_like(sx)
        return _OccBatch(
            occ0=occ,
            p_start=torch.stack([sx, sy, z], dim=-1),
            p_stop=torch.stack([ex, ey, z], dim=-1),
            angle=angle,
            raw=self.raw, H=occ.shape[-2], W=occ.shape[-1],
        )


def _undecorated(method):
    """The function `torch.no_grad()` wrapped, so the adapter runs the
    predictor's OWN forward with grad-mode left on instead of reimplementing
    it.  `torch.no_grad.__call__` uses `functools.wraps`, so `__wrapped__` is
    the original function."""
    fn = getattr(method, "__wrapped__", None)
    if fn is None:
        raise TypeError(f"{method!r} is not a torch.no_grad()-decorated function; "
                        "if the decorator was removed, call it directly instead.")
    return fn


class PredictorGradientAdapter(OccupancyGradientAdapter):
    """Wraps any `Baselines/*` predictor whose `predict_occ(batch)` is a
    `@torch.no_grad()`-decorated, otherwise fully differentiable forward
    (`NFDPredictor`, `WarpedNFDPredictor`, `ResidualNFDPredictor`,
    `ResidualWarpedNFDPredictor`)."""

    def __init__(self, name, predictor, device="cuda", goal_shape="corner"):
        super().__init__(name, device, goal_shape)
        self.predictor = predictor
        self.predictor.model.to(self.device).eval()
        for p in self.predictor.model.parameters():
            p.requires_grad_(False)
        self._forward = _undecorated(type(predictor).predict_occ)

    def predict_step(self, occ, act):
        out = self._forward(self.predictor, self._batch(occ, act))
        return out.to(occ.device)


class SwitchedLinearGradientAdapter(OccupancyGradientAdapter):
    """Pixel-space switched-linear operators (`weights/MODEL-0001-*`,
    `Baselines/LinearForesight/runs/operators_res*.pt`).

    `gate="soft"` uses `Baselines.LinearForesight.model.predict_switched_soft`
    -- a RELAXATION of the fitted model (see that function's note), required
    because the hard `torch.bucketize` gate carries no length gradient.
    `gate="hard"` reproduces the fitted model exactly and is what every
    existing register row was measured with."""

    def __init__(self, name, ckpt, device="cuda", goal_shape="corner",
                 res: int = 32, scale: float = 1.0, gate: str = "soft"):
        super().__init__(name, device, goal_shape)
        assert gate in ("soft", "hard")
        self.gate = gate
        self.res = res
        self.scale = scale
        self.bin_edges = ckpt["bin_edges"].to(self.device)
        ops_key = "switched_ops" if "switched_ops" in ckpt else "operators"
        self.ops = [o.to(self.device).float() for o in ckpt[ops_key]]

    def predict_step(self, occ, act):
        from Baselines.LinearForesight.model import (
            predict_switched, predict_switched_soft)
        from fit_linear_foresight import actions_to_pixels
        ws_min = torch.tensor([OCC_BOUNDS["x_min"], OCC_BOUNDS["y_min"]])
        ws_max = torch.tensor([OCC_BOUNDS["x_max"], OCC_BOUNDS["y_max"]])
        H, W = occ.shape[-2], occ.shape[-1]
        s_px, e_px = actions_to_pixels(act, ws_min, ws_max, (H, W))
        length_m = (act[:, 2:4] - act[:, 0:2]).norm(dim=-1)
        fn = predict_switched_soft if self.gate == "soft" else predict_switched
        return fn(self.bin_edges, self.ops, occ, s_px, e_px, length_m,
                  self.res, (H, W), self.scale)


# ── registry: adding a model is ONE entry here ───────────────────────────────

def _nfd(ckpt, channels=3, features=None):
    def f(device, goal_shape):
        from Baselines.NFD.predictor import NFDPredictor
        return PredictorGradientAdapter(
            "x", NFDPredictor(ckpt, channels=channels, features=features), device, goal_shape)
    return f


def _nfd_warped(ckpt, plate_mode="canonical", wall_channel=False, canon_res=None, scale=1.0):
    def f(device, goal_shape):
        from model.warped_nfd.predictor import WarpedNFDPredictor
        return PredictorGradientAdapter(
            "x", WarpedNFDPredictor(ckpt, plate_mode, wall_channel, canon_res, scale),
            device, goal_shape)
    return f


def _nfd_residual_warped(ckpt, plate_mode="canonical", wall_channel=False,
                         canon_res=None, scale=1.0):
    def f(device, goal_shape):
        from model.residual_nfd.predictor import ResidualWarpedNFDPredictor
        return PredictorGradientAdapter(
            "x", ResidualWarpedNFDPredictor(ckpt, plate_mode, wall_channel, canon_res, scale),
            device, goal_shape)
    return f


def _nfd_residual_unwarped(ckpt):
    def f(device, goal_shape):
        from model.residual_nfd.predictor import ResidualNFDPredictor
        return PredictorGradientAdapter("x", ResidualNFDPredictor(ckpt), device, goal_shape)
    return f


def _retrieval_nfd_ref(ckpt, zero_ref=False):
    """EXP-0059 section 8: NFD with a retrieved reference (`model/
    retrieval_nfd/predictor.py::RetrievalRefPredictor`). `zero_ref=True` is
    the "test-time zeroed reference" control (iii): same checkpoint, donor
    channels forced to 0 at inference."""
    def f(device, goal_shape):
        from model.retrieval_nfd.predictor import RetrievalRefPredictor
        return PredictorGradientAdapter(
            "x", RetrievalRefPredictor(ckpt, zero_ref=zero_ref), device, goal_shape)
    return f


def _switched(ckpt_path, res=32, gate="soft"):
    def f(device, goal_shape):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return SwitchedLinearGradientAdapter("x", ck, device, goal_shape,
                                             res=res, gate=gate)
    return f


OCC_ADAPTERS = {
    # id                          factory
    "nfd_3ch":                    _nfd("Baselines/NFD/runs/nfd_3ch/unet_best.pth"),
    "nfd_3ch_randlen":            _nfd("Baselines/NFD/runs/nfd_3ch_randlen/unet_best.pth"),
    "nfd_3ch_finetuned":          _nfd("weights/MODEL-0003-nfd-multistep-finetuned/checkpoint.pth"),
    "nfd_warped_randlen":         _nfd_warped("Baselines/NFD/runs/nfd_warped_randlen/unet_best.pth"),
    "nfd_residual_warped":        _nfd_residual_warped(
        "Baselines/NFD/runs/nfd_residual_warped_L20mm_pilot_2/unet_best.pth"),
    # randlen-trained NFD variants (same checkpoints as eval_report.MODELS), added
    # 2026-09-24 for the DS-0005 state-superiority analysis (TODO G3a)
    "nfd_warped_randlen_flipaug": _nfd_warped("Baselines/NFD/runs/nfd_warped_randlen_flipaug/unet_best.pth"),
    "nfd_warped_randlen_flipaug_epoch30": _nfd_warped(
        "Baselines/NFD/runs/nfd_warped_randlen_flipaug/unet_epoch_30.pth"),
    "nfd_residual_warped_flipaug_randlen": _nfd_residual_warped(
        "Baselines/NFD/runs/nfd_residual_warped_flipaug_randlen/unet_best.pth"),
    "nfd_residual_worldframe_noaug_ep43": _nfd_residual_unwarped(
        "Baselines/NFD/runs/nfd_residual_unwarped_noaug_randlen/unet_best.pth"),
    "linear_switched_soft":       _switched("weights/MODEL-0001-stage2-visual-switched/checkpoint.pt"),
    "linear_switched_hard":       _switched("weights/MODEL-0001-stage2-visual-switched/checkpoint.pt",
                                            gate="hard"),
}


# EXP-0036 training seeds of the world-frame NFD baseline: registered only once
# their checkpoint exists, so the registry (and its tests) never point at a file
# that has not been trained yet.
import os as _os
for _k in (1, 2, 3):
    _ck = f"Baselines/NFD/runs/nfd_3ch_randlen_seed{_k}/unet_best.pth"
    if _os.path.exists(_ck):
        OCC_ADAPTERS[f"nfd_3ch_randlen_seed{_k}"] = _nfd(_ck)

# Overnight 2026-09-25 narrow-domain models (n20 single layer, 20 mm perpendicular; EXP-0053),
# registered once trained.
for _name, _feat in (("nfd_3ch_narrow_l20", None), ("nfd_3ch_narrow_l20_wide", [16, 32, 64])):
    _ck = f"Baselines/NFD/runs/{_name}/unet_best.pth"
    if _os.path.exists(_ck):
        OCC_ADAPTERS[_name] = _nfd(_ck, features=_feat)
for _res in (32, 64):
    _ck = f"Baselines/LinearForesight/runs/operator_narrow_l20_res{_res}.pt"
    if _os.path.exists(_ck):
        OCC_ADAPTERS[f"linear_narrow_l20_res{_res}"] = _switched(_ck, res=_res)

# EXP-0059 clean-data re-collection (2026-09-28): v2 (ISS-010-fix sampler, train_v2/DS-0015)
# narrow-domain models, same recipe as the v1 (nfd_3ch_narrow_l20 / linear_narrow_l20_res{32,64})
# entries above, trained on the clean corpus for a fair v1-vs-v2 / retrieval comparison.
if _os.path.exists("Baselines/NFD/runs/nfd_3ch_narrow_l20_v2/unet_best.pth"):
    OCC_ADAPTERS["nfd_3ch_narrow_l20_v2"] = _nfd("Baselines/NFD/runs/nfd_3ch_narrow_l20_v2/unet_best.pth")
for _ep in (10, 20, 30, 40, 50, 60):
    _ck = f"Baselines/NFD/runs/nfd_3ch_narrow_l20_v2/unet_epoch_{_ep}.pth"
    if _os.path.exists(_ck):
        OCC_ADAPTERS[f"nfd_3ch_narrow_l20_v2_epoch{_ep}"] = _nfd(_ck)
for _res in (32, 64):
    _ck = f"Baselines/LinearForesight/runs/operator_narrow_l20_v2_res{_res}.pt"
    if _os.path.exists(_ck):
        OCC_ADAPTERS[f"linear_narrow_l20_v2_res{_res}"] = _switched(_ck, res=_res)
# Two more training seeds of the v2 NFD (coordinator, 2026-09-28), same recipe/config as
# nfd_3ch_narrow_l20_v2 (seed 0 == the original unseeded run), for a 3-seed training-noise floor
# on the clean corpus -- mirrors the nfd_3ch_randlen_seed{1,2,3} pattern above.
for _seed in (1, 2):
    _ck = f"Baselines/NFD/runs/nfd_3ch_narrow_l20_v2_seed{_seed}/unet_best.pth"
    if _os.path.exists(_ck):
        OCC_ADAPTERS[f"nfd_3ch_narrow_l20_v2_seed{_seed}"] = _nfd(_ck)

# EXP-0059 section 8: NFD with a retrieved reference + controls, registered
# once each is trained.
_RETRIEVAL_NFD_CK = "model/retrieval_nfd/runs/retrieval_nfd_ref/unet_best.pth"
_RETRIEVAL_NFD_RANDOM_CK = "model/retrieval_nfd/runs/retrieval_nfd_random_donor/unet_best.pth"
_RETRIEVAL_NFD_NOREF_CK = "model/retrieval_nfd/runs/retrieval_nfd_noref/unet_best.pth"
if _os.path.exists(_RETRIEVAL_NFD_CK):
    OCC_ADAPTERS["retrieval_nfd_ref"] = _retrieval_nfd_ref(_RETRIEVAL_NFD_CK)
    OCC_ADAPTERS["retrieval_nfd_ref_zeroed"] = _retrieval_nfd_ref(_RETRIEVAL_NFD_CK, zero_ref=True)
if _os.path.exists(_RETRIEVAL_NFD_RANDOM_CK):
    OCC_ADAPTERS["retrieval_nfd_random_donor"] = _retrieval_nfd_ref(_RETRIEVAL_NFD_RANDOM_CK)
if _os.path.exists(_RETRIEVAL_NFD_NOREF_CK):
    # no-reference twin (coordinator 07:34): donor channels always zero in
    # TRAINING (dataset dropout_p=1.0) AND eval (zero_ref=True here too) --
    # the fair "ref vs no-reference at all" control the earlier random-donor
    # comparison could not isolate (random-donor's noisy-but-present channels
    # may simply hurt, which is not the same claim as "reference helps").
    OCC_ADAPTERS["retrieval_nfd_noref"] = _retrieval_nfd_ref(_RETRIEVAL_NFD_NOREF_CK, zero_ref=True)


def make_occ_adapter(model_id: str, device: str = "cuda", goal_shape: str = "corner"):
    """Return the `OccupancyGradientAdapter` registered under `model_id`.

    This is the live-gradient counterpart of
    `Baselines/common/eval_report.py`'s `MODELS` dict: adding a model is one
    entry in `OCC_ADAPTERS`."""
    if model_id not in OCC_ADAPTERS:
        raise NotImplementedError(
            f"No occupancy gradient adapter registered for '{model_id}'. "
            f"Known: {sorted(OCC_ADAPTERS)}")
    ad = OCC_ADAPTERS[model_id](device, goal_shape)
    ad.name = model_id
    return ad


def assert_dv_convention(adapter, tol: float = 1e-9) -> None:
    """Assert, in code, that `adapter.dv` is `value(after) - value(before)`
    AND that the adapter's declared `higher_is_better` agrees with what its
    value function actually does, so a flipped sign fails loudly here instead
    of inverting every conclusion downstream.

    The test is built from goal GEOMETRY, independent of the value function:
    a single-pixel occupancy at the distance field's minimum (`near`, on the
    goal) against one at its maximum (`far`). Moving mass far -> near is an
    improving push under every value function in this repo, so its `dv` must
    be `< 0` for a COST (`lyapunov`) and `> 0` for a VALUE (`mass_in_region`,
    `signed_mass_in_region`). The adapter must declare `value_fn` and
    `higher_is_better`, and the declared sense must match
    `Baselines.common.goals.higher_is_better_for(value_fn)`
    (experiments/METRICS.md, SIGN section; TODO M4)."""
    from Baselines.common.goals import higher_is_better_for, improvement
    value_fn = getattr(adapter, "value_fn", None)
    hib = getattr(adapter, "higher_is_better", None)
    assert value_fn is not None and hib is not None, (
        "adapter must declare `value_fn` and `higher_is_better`")
    assert hib == higher_is_better_for(value_fn), (
        f"adapter declares higher_is_better={hib} but value_fn={value_fn!r} "
        f"is registered as higher_is_better={higher_is_better_for(value_fn)}")
    g = OCC_GRID
    dev = adapter.device
    near = torch.zeros(1, g, g, device=dev)
    far = torch.zeros(1, g, g, device=dev)
    dwf = adapter.dw.reshape(-1)
    lo = int(torch.argmin(dwf)); hi = int(torch.argmax(dwf))
    near[0, lo // g, lo % g] = 1.0
    far[0, hi // g, hi % g] = 1.0
    dv = float(adapter.value(near)) - float(adapter.value(far))
    gain = improvement(dv, 0.0, higher_is_better=hib)
    assert gain > tol, (
        f"dv convention broken for {value_fn} (higher_is_better={hib}): moving mass "
        f"onto the goal gave dv={dv:+.6g}, which reads as NOT an improvement")
