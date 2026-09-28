# Architecture

This is the repository map and extension guide. Data contracts and coordinate conventions are documented in `INTERFACES.md`. Utility ownership and transform guidance are documented in `UTILITIES.md`. The Genesis-as-model sampling MPC (CEM/MPPI ceiling baseline) has its own reference doc, `oracle_mpc_design.md`, since its design has enough moving parts (batched multi-env rollout, snapshot/restore state management, sampling optimizers) to warrant a dedicated write-up; this file only maps where its pieces live. The human-piloted variant of that same ceiling baseline (a person picks the action each step instead of CEM/MPPI, refined by a local grid search) has its own doc, `human_demo_design.md`, for the same reason. `scaling_to_200_objects.md` is the reference for the simulator's non-obvious settings — why the solver budget, contact-pair cap, settling criterion, particle spawn and pusher-plate actuator model hold their current values, and what large piles cost — and should be consulted before changing anything under `Genesis/configs/` or `SandboxManipulation`'s physics setup. `plate_model.md` covers the pusher tool specifically — its actuator model, the three phases of a push, and `reaction_report()`, which gives the peak actuator force, torque and granular reaction per action for setting real-robot limits. `genesis_world_migration.md` records the move from Genesis 0.4.5 to Genesis World 1.3.3 and the breaking changes it required.

> **Documentation is part of the change, not a follow-up.** Any change to a
> major information flow, a module's responsibility, a default value, or an
> added feature must update the relevant doc(s) below — `ARCHITECTURE.md`
> (module map, philosophy), `INTERFACES.md` (data contracts), `UTILITIES.md`
> (utility ownership), `oracle_mpc_design.md` (oracle MPC internals) — in the
> same change that makes the change, not as a separately-requested pass
> afterward. See `.claude/skills/project-overview/SKILL.md` for the
> project-level pointer to all of these.

## Design Philosophy

Recurring patterns worth preserving when extending this codebase:

- **Stateless transforms, stateful wrappers.** `transforms/functional.py` holds
  pure, dependency-light conversion functions (particle↔occupancy, action↔camera
  coordinates); heavier stateful classes (`model/eulerian_wrapper.py`,
  `simple_mpc/`) call into it, never the other way around. New geometry/representation
  conversions belong in `transforms/`, not copy-pasted into whichever wrapper needs them first.
- **Adapter pattern decouples the optimizer from the model.** `simple_mpc/mpc.py`'s
  gradient-descent loop never branches on model type; `simple_mpc/adapters.py`
  exposes a uniform `obs_to_state` / `predict_step` / `compute_reward` surface
  (INTERFACES.md §3.4) per model family. `simple_mpc/oracle_mpc.py` has no
  adapter at all — the "model" is the Genesis simulator itself, so there is
  nothing to abstract over — but it reuses the same loss registry and occupancy
  utilities as the adapters do, so the two MPC variants stay comparable rather
  than diverging into parallel implementations.
- **One loss registry serves both training and MPC cost.** `training/losses.py`
  losses are the single source of truth for "how good is this occupancy
  relative to a target" — used with a batch-mean reduction during training and
  a `per_sample=True` reduction when ranking MPC candidates. A loss added for
  one use case is available to the other for free; don't hand-roll a
  cost-specific duplicate of a training loss.
- **The simulator core stays purpose-agnostic.** `Genesis/sandbox_manipulation_clean.py`
  exposes generic hooks (`execute_action(..., on_phase=callback)`,
  `set_particle_state`/`broadcast_state_from_env`) rather than baking in
  camera, video, or MPC-specific logic. Anything that needs a camera or cares
  about *why* a push is happening lives one layer up, in the env wrappers
  (`env/genesis_env.py`, `simple_mpc/genesis_oracle.py`) or MPC loops — never
  in the simulator wrapper itself. `push_and_record`/`flush_transitions`
  (automatic MPC transition recording, `Genesis/transition_buffer.py`) are the
  one exception the core owns directly rather than exposing as a hook —
  justified because recording only touches state the core already tracks
  (`_particle_state`) and needs zero downstream-specific knowledge (no
  camera, no reward, no "MPC step" concept beyond an opaque tag the caller
  supplies) — it's generic bookkeeping, not a purpose-specific feature.
- **Shared utilities over duplicated logic.** Cross-cutting operations used by
  more than one MPC variant or script (`write_video_frame`,
  `particles_to_occupancy`, `footprint_radius_voxels`, `OccupancyReward`) live
  in `utils.py` / `transforms/functional.py` / `simple_mpc/occupancy_reward.py`
  and are imported, not re-derived. If you find yourself copying a snippet a
  second time, that's the signal to extract it instead.
- **Config-driven defaults.** Hyperparameters and physical constants live in
  YAML (`simple_mpc/config/*.yaml`), not hardcoded in Python, so swapping
  models, optimizers, or physical setups doesn't require code edits.
- **Docs mirror code boundaries.** Each doc file's scope matches a real code
  boundary (module map, data contracts, utility ownership, one subsystem's
  design) rather than being a single unstructured wiki page — this is what
  makes "update the docs" a small, targeted diff instead of a chore, and why
  it belongs in the same change as the code.

## Module Map

```
physics/
  normalization.py      PhysicsBounds (canonical raw<->normalized mapping)

registry/
  model_registry.py     register_model, build_model, ModelTrainingWrapper
                        (+ EulerianTrainingWrapper, LagrangianTrainingWrapper)
  dataset_registry.py   register_dataset, build_dataset, EulerianDatasetWrapper,
                        LagrangianDatasetWrapper, GenesisParticlePushDataset

model/
  model_card.py         ModelCard save/load, load_model_from_card (MPC wrapper),
                        load_net_from_card (bare network for eval scripts)
  NFDUNetFilm.py        NFDUNetFiLM / NFDUNetFiLMShallow — the U-Net
                        implementations built by the "unetfilm*" factories
  eulerian_wrapper.py   EulerianModelWrapper (MPC-facing occupancy wrapper),
                        heuristic push models (splat/spread/spread2/cumulative/
                        fluid) plus register_push_model/build_push_model — a
                        parallel, checkpoint-free registry (these have no
                        learned weights, so they don't go through
                        registry/model_registry.py or model_card.py),
                        UNetFiLMPushModel, particle<->occupancy helpers
  diff_mass_push.py     differentiable mass-push kernels used by the
                        heuristic push models
  gnn_dyn.py            PropNetDiffDenModel (Lagrangian GNN dynamics)
  NCAModels.py          NCAWithPhysics       → model type "nca"
  SpatTransNet.py       EulerianSTN          → model type "spatial-transformer"
  UNetModels_modular.py UNet (config-driven) → model type "unet-modular"
                        (these three now live at `model/` top level, not
                        under `futureintegration/` — moved out when each was
                        registered; the rest of `futureintegration/` await
                        consolidation/are skipped/broken, see its README)
  futureintegration/    salvaged architectures not yet promoted to the top
                        level — see its README for the breakdown
  retrieval/            EXP-0059: retrieval-based transition model groundwork
                        (`docs/experimental_design/retrieval_based_modeling.md`).
                        Genesis-free (torch/numpy/scipy only). A PARTICLE-IN/
                        PARTICLE-OUT contract (`predict_particles(states0,
                        p_start, p_stop) -> states1`), deliberately NOT the
                        occupancy-in/out contract `simple_mpc.adapters
                        .OCC_ADAPTERS` uses — a retrieval model needs the
                        query's raw cube positions to retrieve against, which
                        an occupancy grid has already discarded.
    frame.py                 metre-space push-frame canonicalisation of raw
                            (x, y) positions and cube yaws (origin at the
                            push START, +u along the push direction, +v
                            lateral) — the object-centred analogue of
                            `transforms.functional.push_frame_transform`,
                            which is image-centred (origin at the push
                            MIDPOINT) and only warps rasterised occupancy.
                            `tray_corners_push_frame` gives the tray walls in
                            the same per-transition frame.
    bank.py                  `TransitionBank` — a flat, per-object transition
                            database (`D = {(s_i, a_i, s'_i, Delta_i)}`) built
                            from DS-0008 (+DS-0010), canonicalised into every
                            transition's own push frame; `moved` per-object
                            mask (displacement threshold, a parameter) and
                            `moved_count_histogram` for the sanity stat.
                            Save/load persists it as one `.pt` file.
                            `load_curated` (2026-09-28, DS-0014) is the
                            preferred loader going forward: reads the
                            curated bank (interaction-set mask `in_set` +
                            ISS-010 touchdown-legality flag `legal` per
                            row), excluding illegal rows by default —
                            `.build()`/`.from_states()` are UNCHANGED (still
                            what `model/retrieval_nfd/donors.py` asserts its
                            own bank matches row-for-row).
    distance.py               `DistanceConfig` — a configurable windowed
                            symmetric push-frame chamfer distance (optional
                            cap + separate mismatch penalty for "no
                            plausible match", corridor upweighting for
                            cubes the plate actually sweeps vs merely
                            nearby, optional wall-distance feature) and
                            `topk_search`: an EXHAUSTIVE, GPU-batched top-k
                            search over the whole bank (chunked over the
                            bank dimension so a `(query, bank_chunk, n, n)`
                            tensor stays bounded regardless of bank size —
                            see `bank.py`'s `MAX_RECOMMENDED_BANK_SIZE`).
                            `bank_valid`/query `valid` masks stop a
                            transition with nothing in its own window from
                            winning on a spurious zero-weight "0 distance".
    predictor.py             `PersistencePredictor` (states1=states0; the
                            reference floor — accuracy=0 by construction of
                            `eval_narrow.py`'s own `acc()`) and
                            `RetrievalPredictor` (configurable whole-bank
                            k-NN via `distance.py`; aggregations `nn1` /
                            `cube_median` / `cube_weighted_mean` /
                            `occ_mean` / `occ_weighted_mean`; a Hungarian
                            cube correspondence transfers each neighbour's
                            push-frame displacement/yaw-delta, gated by the
                            donor cube's own `moved` flag). **FIXED
                            2026-09-28** (a real bug, not a toggle): the
                            match used to run over ALL n query/donor cubes
                            with no distance limit, so a query cube far from
                            everything could inherit a large,
                            physically-nonsensical displacement from
                            whatever donor cube it was force-paired with
                            (`retrieval_debug.py`'s `q0908` figure). Now both
                            sides are restricted to their own geometric
                            interaction set (`interaction.py`, read from
                            `bank.in_set` when curated, else every cube is
                            eligible — old behaviour) AND every matched pair
                            is gated by `distance_gate` (default 6mm) —
                            farther apart transfers nothing, the cube stays.
                            `NearestTransitionPredictor` is a thin k=1/nn1
                            backward-compatible alias.
                            **Hedged occupancy is scoring-only** (2026-09-28
                            user guidance): `predict_particles` ALWAYS
                            returns one CLEAN, committed-to state — the
                            `occ_mean`/`occ_weighted_mean` aggregations'
                            hedge is exposed ONLY via the optional
                            `predict_occ` method (same `hasattr` pattern as
                            `WarpedNFDPredictor.predict_occ_canonical`), and
                            only `eval_retrieval.py`'s single-step
                            `accuracy_1` uses it — rollout and `slateN`
                            always call `predict_particles`. Per-cube
                            overlap is NOT resolved (documented limitation).
                            Every predictor also exposes
                            `predict_particles_with_confidence` — the clean
                            prediction plus `top1_dist`/`knn_disagreement`
                            computed from the SAME search (no extra cost),
                            for finding actions the bank predicts least
                            well, not for scoring.
    val_split.py              leakage-safe DS-0008 train/validation split
                            for hyperparameter tuning, grouped by CHAIN
                            (`(file_index, chain_env)`, never row-random —
                            a chain's 8 steps are near-duplicates) so tuning
                            never touches DS-0009. Returns a tuning
                            `TransitionBank` (held-in DS-0008 chains + all
                            of DS-0010) and the held-out chains repackaged
                            in `eval_retrieval.py`'s chain-dict shape.
                            DS-0008 has no same-state action pools, so
                            `slateN` cannot be validated this way — only
                            `accuracy_1`/`rollout_accuracy_4` can; `slateN`
                            is reported as a DS-0009 TEST number only, for
                            the shortlisted best configs
                            (`sweep_retrieval.py`).
    interaction.py            (2026-09-28, coordinator follow-up B) `interaction_set` — a
                            TRUTH-FREE geometric "affected set" per transition (design doc
                            section 6.1): cubes the blade sweeps + a forward contact-chain
                            closure, batched (no ground truth needed, so a query — including a
                            rollout step with no future — can always compute it). Tuned on
                            training data (tau=12mm, angle_max_deg=60) to reach ~96% recall of
                            the bank's own truth `moved` set at ~90% precision. `recall_precision`
                            is the small pooled-over-cubes scorer that tuning used.
  retrieval_nfd/        EXP-0059 section 8: "NFD with a retrieved reference"
                        -- an IMAGE-space model (occupancy-in/out, plugs into
                        `simple_mpc.adapters.OCC_ADAPTERS` like every other
                        NFD variant), NOT the particle-in/out `retrieval/`
                        package above. The narrow NFD's UNet
                        (features [4,8,16], residual head) with 2 extra
                        input channels: a k=1 (eval) / random-of-top-3
                        (train) retrieved donor transition's before/after
                        cubes, mapped from the DONOR's push frame into the
                        QUERY's push frame then world, and rasterised with
                        the SAME cv2-box rasteriser (`render_cube_boxes_batch`,
                        reimplementing `Genesis.training.dataset.PileSweepData
                        ._draw_particle_grid`'s box path) that produces channel
                        0, which itself comes directly from the REAL
                        `Baselines.NFD.nfd_lib.PileSweepData3Ch` -- bit-exact
                        with the narrow NFD's own training data (a from-scratch
                        soft-cube approximation was tried and reverted once the
                        donor SEARCH was fixed to be fast enough that the exact
                        rasteriser fit the time budget). Everything is
                        precomputed ONCE into a flat cache
                        (`cache/retrieval_ref_cache.pt`) -- training reads no
                        raw corpus and does no retrieval search.
    render.py                 `render_cube_boxes_batch`: the cv2-box
                            rasteriser, batched over rows in a Python loop
                            (cv2 has no batched primitive; measured ~2ms/row).
    donors.py                 also `build_query_rows_from_pilesweepdata`,
                            which builds channels 0-2/target + raw states
                            directly off `PileSweepData3Ch` (bit-exact,
                            not re-derived); the donor top-k search is fully
                            vectorised (chain keys -> integer ids, boolean
                            mask + stable argsort per chunk -- the earlier
                            per-row `.tolist()` Python loop was the actual
                            ~70-minute bottleneck, not the chamfer search).
    donors.py                 reloads DS-0008+DS-0010 with a parallel
                            per-row CHAIN KEY (file+chain_env / source_file)
                            `model/retrieval/bank.py::TransitionBank` itself
                            does not keep, so donor search can exclude the
                            query's own chain; the frozen EXP-0059 R1
                            sweep-chosen retrieval key; chain-excluding
                            top-k search and a corridor-cube-count-bucketed
                            random donor for the control model.
    precompute.py             one-time cache builder; runs the MANDATORY
                            unit tests inline (query's own cubes through the
                            donor-render path reproduce occ0; the push-frame
                            v-mirror round-trips against an independently
                            derived world-space reflection).
    dataset.py / lib.py       flat cache-backed `Dataset` + its
                            `nfd-genesis-retrieval-ref` dataset-registry
                            entry (`registry.dataset_registry
                            .EulerianDatasetWrapper` contract); reuses
                            `Baselines.NFD.nfd_lib`'s existing generic
                            `nfd-unet3ch` model factory unchanged
                            (`in_channels: 5`) -- no new model type.
    predictor.py              `RetrievalRefPredictor`, matching
                            `Baselines/NFD/predictor.py::NFDPredictor`'s
                            `predict_occ(batch)` contract, PLUS
                            `predict_step_particles(occ0, act, states0)`,
                            which `eval_extended.py::eval_occ_model` now
                            dispatches to (patched, 3 call sites) whenever a
                            TRUE particle state is available -- pseudo-cube
                            extraction from `occ0` via connected-components
                            centroids is now only a fallback for rollout
                            steps after the first. `zero_ref=True` is the test-time
                            zeroed-reference control.
  warped_nfd/           "warped NFD": the NFD UNet baseline predicting in
                        the canonical PUSH FRAME instead of the world frame
                        (a new model family, not a minor parameter change
                        over the NFD baseline in Baselines/NFD/ — see the
                        Baselines/NFD/ entry below for the split rationale).
    lib.py                  registers dataset `nfd-genesis-3ch-warped`
                            (PileSweepData3ChWarped, adds a `push_px` batch
                            key) and model `nfd-unet-warped`
                            (WarpedNFDWrapper; config knobs plate_mode:
                            canonical|warp, wall_channel, canon_res, scale)
    predictor.py            eval-time twin: `build_canonical_stack` (the ONE
                            canonical-frame input-stack builder, shared
                            verbatim by lib.py's WarpedNFDWrapper and this
                            module's WarpedNFDPredictor so training/eval
                            cannot drift apart — train/eval parity measured
                            at ~1e-6 world-frame prob diff), WarpedNFDPredictor,
                            build_predictor_warped[_walls]. Genesis-free by
                            design (does not import Genesis.training.dataset
                            / the dataset registry), matching the rest of
                            `model/`.
    WARPED_NFD_NOTES.md     coordinate-convention traps ((col,row) vs
                            (row,col) point order) and the plate_mode timing
                            derivation
  residual_nfd/         EXP-0022 R1/R2: image-space RESIDUAL
                        PARAMETERISATION for NFD (not a residual objective —
                        the loss stays `eulerian_combined` world-frame MSE
                        against the absolute `occ1` target, unchanged).
    lib.py                  registers model `nfd-unet3ch-residual` (R1,
                            unwarped control; reuses the existing
                            `nfd-genesis-3ch` dataset unchanged) and
                            `nfd-unet-warped-residual` (R2, warped; reuses
                            `nfd-genesis-3ch-warped` and
                            `warped_nfd/predictor.py::build_canonical_stack`,
                            unchanged)
    predictor.py            eval-time predictors registered in
                            Baselines/common/eval_report.py's `MODELS` as
                            `nfd_residual_unwarped_L20mm_pilot`/
                            `nfd_residual_warped_L20mm_pilot`
  flow_nfd/             EXP-0025: flow/advection field prediction — a
                        per-pixel BACKWARD displacement field warped through
                        `grid_sample`, mass-conserving and residual by
                        construction.
    lib.py                  registers model `nfd-flow-warp`; reuses the
                            existing `nfd-genesis-3ch` dataset unchanged
    predictor.py            eval-time twin (duplicates the flow/warp math
                            so eval and train agree, same pattern
                            `residual_nfd/predictor.py` uses)
    supervised.py           direct particle-correspondence supervision for
                            the flow head (`build_flow_targets_for_split`,
                            the x8-augmentation-with-vector-rotation helper
                            `augment_flow_batch_x8`), on top of the
                            photometric loss above

fit_linear_foresight.py  fits and falsifies the switched-linear pixel operator
                        of Suh & Tedrake 2020 on a single-push-length dataset;
                        persistence / identity-operator / heuristic baselines,
                        swept-region metric, ridge sweep, non-negative fit.
                        Results: reports/linear_foresight_report.md

transforms/
  particle_fields.py    particle set -> grid projections (points_to_density /
                        points_to_heightmap / points_to_mask) plus
                        fraction_in_bounds, the conservation check. Out-of-bounds
                        points are DROPPED, not clamped, so escaped material
                        cannot pile onto the boundary cells and invent mass.
                        UNCLAMPED and mass-preserving, unlike
                        particles_to_occupancy, which clamps to a binary
                        silhouette and would discard the depth that is the whole
                        point of a continuum. Pure torch, GPU-free tests.
  functional.py         particles_to_occupancy (+ footprint_radius hard-disk
                        splat, shape_factor for non-spherical particles),
                        footprint_radius_voxels, draw_plate_soft,
                        genesis_action_to_cam3d, genesis_particles_to_cam3d,
                        build_action_delta, action_to_pose (shared 4D-derived-
                        yaw / 5D-explicit-yaw action convention — see
                        docs/human_demo_design.md)
  representation.py     Compose, EnsureRepresentation, occupancy/particle
                        alias transforms, build_transforms

training/
  types.py              TrainingBatch and ModelOutput contracts
  losses.py             register_loss, build_loss, EulerianCombinedLoss
                        (per_sample=True reduction mode for MPC-candidate
                        costs, in addition to the scalar training reduction),
                        ScoreMapWeightedLoss (loss = -occupancy·score_map, so
                        MPC optimization can share the exact reward-reporting
                        objective), LagrangianMSELoss
  metrics.py            build_metrics, EulerianMetrics, LagrangianMetrics
  trainer.py            Trainer loop and config include resolution
  train.py              CLI entry point

simple_mpc/
  mpc.py                run_simple_mpc gradient-descent MPC loop (learned/
                        heuristic models, via adapters)
  adapters.py           EulerianAdapter, GNNAdapter, make_adapter
                        (the adapter surface defined in INTERFACES.md §3.4)
  learned_mpc.py        closed-loop MPC with a LEARNED model as the objective under
                        a fixed wall-clock budget per decision: planners rank / gd
                        (projected Adam, restarts) / cem / mppi over
                        OCC_ADAPTERS models; progress scored with soft truth
                        (occ_for_scoring); project_push = the 20-70 mm / 4 mm
                        legality projection (EXP-0023's). Executor-agnostic
                        (`execute` callable), per-step checkpoint hook. TODO G1b
  value_functions.py    capacity-aware MPC value functions (sliced EMD to the
                        uniform goal target, linearised-EMD distance field,
                        over-capacity repulsion) + coverage metrics/ceiling;
                        used via ModelObjective(value=...) (EXP-0055)
  goal_aware_sampling.py goal-aware selection of the planner's starting
                        candidates from a pile-aware bank (carry model,
                        misplaced mass, deposit quality, OT proposals) (EXP-0056)
  gt_bank.py            GroundTruthBank -- append-only store of simulated push
                        OUTCOMES (final particle states), keyed by execution
                        path (SIM_PATHS) + start-state hash + exact action
                        bytes, so any goal/value fn can be re-scored without
                        Genesis. Genesis-free (evaluate() takes a simulate
                        callable). Payload: datasets/DS-0004-ground-truth-bank/
  action_sampler.py     candidate-action samplers (uniform, physics-aware,
                        collision-aware, OT-guided)
  ot_planner.py         OTPlannerSparse (Sinkhorn OT action initializer)
  occupancy_reward.py   OccupancyReward (goal-mask scoring: compute_score_tensor
                        for the reward map, goal_occupancy_mask for a plain
                        binary loss target)
  benchmark.py          adapter/push throughput benchmarks
  debug_vis.py          MPC debug visualization panels/videos
  genesis_oracle.py     GenesisOracleEnv — batched multi-env Genesis wrapper
                        for oracle MPC (see docs/oracle_mpc_design.md).
                        step()/rollout_candidates() record via
                        push_and_record (real steps flush immediately; tagged
                        is_candidate=True planning rollouts accumulate until
                        the next real step); set_recording_context(...) tags
                        an episode's flushes before it runs
  oracle_mpc.py         run_oracle_mpc, load_oracle_config — sampling MPC
                        where the Genesis simulator itself is the prediction
                        model; no adapter needed (see docs/oracle_mpc_design.md)
  sampling_optimizers.py CEMOptimizer, MPPIOptimizer, make_sampling_optimizer
                        (shared ask/tell/best skeleton; gradient-free)
  human_grid_search.py  build_action_grid, grid_search_refine — local 5D grid
                        search around a human-drawn action, via the oracle
                        simulator (see docs/human_demo_design.md)
  human_mpc.py          HumanDemoSession (propose/commit/finished/finalize),
                        save_episode — interactive human-piloted episodes
                        over GenesisOracleEnv, parallel to (not built on)
                        run_oracle_mpc's automated loop
  config/               base MPC config, config_oracle*.yaml,
                        config_human_demo.yaml, experiments/

RealData/
  dataset.py            RealPileSweepData (real camera data; mirrors the
                        PileSweepData output format)

configs/
  model/                one YAML per architecture
  dataset/              one YAML per data source
  training/             experiment YAMLs composing model+dataset+training

Genesis/
  sandbox_manipulation_clean.py  SandboxManipulation — low-level multi-env
                        simulator wrapper (build/reset/execute_action/
                        update_material_state). execute_action's on_phase
                        callback (fired at 'post_lower' / 'post_sweep',
                        no-op if omitted), set_particle_state /
                        broadcast_state_from_env, and push_and_record /
                        flush_transitions (automatic transition recording,
                        see below) are generic, purpose-agnostic primitives —
                        reused by env/genesis_env.py,
                        simple_mpc/genesis_oracle.py, and this module's own
                        data_collection_clean.py, not oracle-specific.
  state_library.py       StateLibrary / build_state_library — a bank of
                        *settled* pile states, so a reset is a pose write
                        rather than a re-settle (measured ~6 ms vs ~37 s at
                        200 particles). Settles are amplified by the
                        container's symmetry group (D4, x8, for a square
                        tray), which is why a dozen settles yield a hundred
                        distinct starts. Genesis-free (torch only),
                        unit-tested in tests/test_state_library.py. Opt-in
                        from data_collection_clean.py via --state-library.
  placement_sampling.py  configuration-space sampling of tool touchdown
                        poses that do not collide with the pile: occupancy
                        grid -> rotated-rectangle dilation per yaw bin
                        (Minkowski sum) -> free set, with a Euclidean
                        distance transform for clearance-biased draws.
                        Consumed by SandboxManipulation.generate_action_
                        samples(placement_aware=True), which falls back to
                        the blind draw per sample when the free set is empty.
                        Genesis-free (torch/numpy/scipy), unit-tested in
                        tests/test_placement_sampling.py.
  cube_spectrum_collection.py  piled-cube collection at a range of cube
                        counts, so model performance can be read as a function
                        of particle count. Piles with the pyramid/heap spawn (a
                        dropped pile cannot exceed one layer) and redraws the
                        heap every episode, so start diversity is free. NOTE:
                        cost is set by contact-island size, not cube count --
                        0.36 s/transition at n=20 but 9.93 at n=30, because the
                        heap percolates from several islands into one and Newton
                        factorizes a dense Hessian per island
                        (docs/scaling_to_200_objects.md).
  spawn_geometry.py      stepped-pyramid particle spawn layouts
                        (pyramid_layer_plan / pyramid_positions), pure torch,
                        unit-tested in tests/test_spawn_geometry.py. The only
                        mechanism found that yields a pile more than one layer
                        deep: the dropped spawn leaves 90-94% of particles in
                        layer 0 whatever the extent or friction, because cubes
                        bounce outward on landing. Reached via
                        shuffle_particles(spawn_mode="pyramid") or
                        --spawn-mode pyramid; see
                        docs/linear_foresight_findings.md section 5. The
                        spawn: block's pyramid_gap / pyramid_pos_jitter /
                        pyramid_yaw_jitter / pyramid_lift shape how much it
                        collapses: an unperturbed pyramid is exactly at rest
                        and re-settles byte-identically (measured 0.0 mm), so
                        without a lift every episode starts from the same
                        lattice -- see scripts/probe_pyramid_setups.py and
                        scripts/probe_pyramid_diversity.py.
  action_sampling.py     batch-aware action shaping AND action-space
                        restriction, both pure torch (no `import genesis`),
                        unit-tested in tests/test_action_sampling.py.
                        Shaping: equalize_travel_distance /
                        shared_batch_distance share one push length across a
                        lockstep batch (the sweep is sized from the LONGEST
                        travel, so independent lengths make every env run for
                        the longest one's duration).
                        Pile-aware: pile_contact_starts places the blade one
                        particle-width from the pile's near face, laterally
                        aligned so its swath contains material, so every
                        simulated push starts in contact and sweeps through the
                        pile (docs/piled_collection.md).
                        Restriction: blade_normal / sampling_box /
                        constrain_push / relative_blade_angle implement the
                        perpendicular-push and fixed-push-length constraints
                        that the switched-linear visual-foresight baseline
                        needs (docs/linear_visual_foresight_baseline.md §7),
                        exposed as generate_action_samples(
                        perpendicular_pushes=, push_length=) and as
                        data_collection_clean.py's --perpendicular-pushes /
                        --push-length. ray_box_max_travel is the shared
                        primitive. Where a constrained push would leave the
                        tray the START is moved, never the push length —
                        shortening would silently drop a transition out of
                        its length bin.
                        duplicate_action_mask flags any action that coincides
                        (within a position and circular-heading tolerance)
                        with another in the same batch — used by
                        same_state_slate_collection.py to enforce that a
                        slate's X candidate sequences never repeat the same
                        (start, heading) at a step.
  same_state_slate_collection.py  same-state candidate slates: settle ONE
                        pile, broadcast it IDENTICAL to every env via
                        StateLibrary.apply_per_env (same index repeated), let
                        each env draw its own action, execute, record — so
                        action-ranking results are not confounded by
                        different envs starting from different piles (see the
                        module docstring for the EXP-0008 motivation). Each
                        state is one on-disk batch (`_{k}_data.pt`); a
                        manifest.json maps batch index -> state-library
                        index. --n-steps > 1 collects independent multi-push
                        SEQUENCES (one per env) from the same broadcast start,
                        calling collect_data_samples(n_samples=1, ...) once
                        per step so each step is still its own batch file —
                        the manifest additionally records slate_idx/step_idx/
                        env_count per batch so a sequence can be reassembled.
                        Multi-step collection uses placement_aware=True (NOT
                        pile_aware=True: pile_aware's action-sampling branch
                        returns before placement_aware's ever runs, so they
                        cannot compose — passing both forces pile_aware off
                        with a logged warning) plus perpendicular_pushes and a
                        fixed push_length, and additionally resamples (before
                        simulating, never truncating) any action that is
                        wall-clipped or a within-slate duplicate of another
                        env's (start, heading) — see
                        action_sampling.duplicate_action_mask and
                        _draw_validated_actions.
  transition_buffer.py   TransitionBuffer — accumulates and saves the
                        before/after/action transitions push_and_record
                        records, in the same on-disk format
                        data_collection_clean.py's dataset files use.
                        Genesis-free (no `import genesis`), independently
                        unit-tested.
  data_collection_clean.py  batched random-push dataset collection; the
                        multi-env execute_action(p_start[K,3], p_stop[K,3],
                        angle[K]) pattern that GenesisOracleEnv's rollout
                        batching mirrors. Does not call push_and_record (has
                        its own save path), so transition recording is a
                        no-op here regardless of config. Action-space
                        restrictions (--perpendicular-pushes, --push-length)
                        are recorded in each batch's saved config, so a
                        restricted dataset is identifiable on disk.
  configs/collection_foresight_single_operator.yaml
                        run_collection.py plan for the targeted collection of
                        ONE switched-linear transition operator (one push
                        length, perpendicular, one pile size / material).
  configs/collection_pile30.yaml
                        run_collection.py plan for 30 cubes as a COMPACT
                        MULTI-LAYER pile with pushes that start in contact with
                        it — see docs/piled_collection.md for the spawn
                        (shuffle_particles(pile_extent=, pile_layers=)) and
                        action (generate_action_samples(pile_aware=True)) modes
                        and every flag that activates them. Both default OFF, so
                        existing configs and datasets are unaffected.
  benchmark_n_envs.py    throughput sweep for picking simple_mpc.oracle_mpc's
                        n_envs default; run as `python -m Genesis.benchmark_n_envs`
                        (must run as a module from the repo root — see its
                        docstring)
  training/dataset.py    PileSweepData — raw Genesis sim dataset, wrapped by
                        the "genesis" entry in dataset_registry.py; also
                        loads push_and_record's output (superset schema,
                        extra keys ignored) if pointed at its output directory

env/
  genesis_env.py         GenesisEnv — single-env bridge from
                        SandboxManipulation to the MPC-facing
                        observation/action interface used by
                        simple_mpc/mpc.py. step() records each real push via
                        push_and_record and flushes it to disk immediately
                        (flush_after=True); set_recording_context(...),
                        called once per episode before it runs, tags those
                        per-step flushes with episode identity.

tests/
  test_configurable_unet.py       registry/model smoke tests (pytest)
  test_futureintegration_models.py  nca/spatial-transformer/unet-modular smoke tests
  test_model_card.py             model_card save/load round-trip
  test_push_model_registry.py    build_push_model + EulerianModelWrapper
                                  end-to-end forward for all 5 heuristics
  test_losses_per_sample.py      EulerianCombinedLoss/ScoreMapWeightedLoss
                                  per_sample reduction mode (Genesis-free)
  test_footprint_splat.py        particles_to_occupancy footprint_radius +
                                  shape_factor, genesis_particles_to_cam3d
                                  (Genesis-free)
  test_sampling_optimizers.py    CEM/MPPI convergence, bounds, warm-start
                                  (Genesis-free, synthetic cost function)
  test_transition_buffer.py      TransitionBuffer append/save schema
                                  round-trip (Genesis-free)
  test_action_to_pose.py         transforms.functional.action_to_pose 4D vs
                                  5D branch, batching (Genesis-free)
  test_human_grid_search.py      build_action_grid shape/centering/bounds/
                                  broadcast (Genesis-free)
  test_state_library.py          symmetry-expansion maths for the settled-state
                                  bank: D4 group size, mirrored quaternions are
                                  proper rotations, rigidity (Genesis-free)
  test_placement_sampling.py     C-space free-placement maths: occupancy,
                                  rotated-rectangle dilation, distance
                                  transform, empty-free-set degradation
                                  (Genesis-free)
  scaling_investigation/         archived probes, raw measurements and the
                                  settling diagnosis behind
                                  docs/scaling_to_200_objects.md. NOT part of
                                  the pytest suite — one-shot GPU measurement
                                  scripts, run as
                                  `python -m tests.scaling_investigation.<x>`.
                                  verify_fixes.py and verify_new_features.py
                                  there are the end-to-end assertions worth
                                  re-running after touching
                                  Genesis/sandbox_manipulation_clean.py;
                                  probe_collection_health.py and
                                  record_simulation_video.py are the pair to
                                  run before trusting a long collection —
                                  the first checks a realistic run's output
                                  statistics (goal-reach rate, per-env
                                  settledness, action-space bias, escaped
                                  particles), the second renders it to video
                                  for whatever nobody thought to measure.
                                  Findings from both are in section 8 of
                                  docs/scaling_to_200_objects.md.

utils.py                shared root-level helpers (YAML I/O, action geometry,
                        point-cloud ops, goal-shape generation,
                        write_video_frame) used by model/eulerian_wrapper.py,
                        env/genesis_env.py, and simple_mpc/
```

Entry points beyond `training/train.py`:

```
run_experiments.py       MPC experiment-suite runner; loads trained models via
                         model cards, or heuristic push models via
                         model.eulerian_wrapper.build_push_model (model.type:
                         eulerian, need_weights: false); writes
                         outputs/experiments/ (per-episode/experiment data;
                         real-step transitions additionally flow to the
                         shared Genesis/data/mpc_runs/ pool — see UTILITIES.md §1.2)
run_experiment_batch.py  subprocess driver running run_experiments.py over
                         multiple suite YAMLs
run_oracle_mpc.py        Oracle (Genesis-as-model) MPC entry point — builds
                         one GenesisOracleEnv, runs episodes.n_episodes
                         episodes, saves full per-episode trajectories
                         (raw frames, point clouds, predicted-vs-actual
                         occupancy, video) — see docs/oracle_mpc_design.md
human_mpc_gui.py         Human-demonstration GUI over the same
                         GenesisOracleEnv — drag-and-drop action input,
                         local grid-search refinement, full multi-step
                         episodes saved in run_oracle_mpc.py's schema — see
                         docs/human_demo_design.md
debug_mpc_gui.py         Interactive learned/heuristic-model MPC debugger
                         (single-env GenesisEnv, gradient-descent action
                         refinement) — human_mpc_gui.py reuses its
                         tile-image helpers and canvas-drag interaction
visualize.py             dataset / occupancy / prediction visualization

Baselines/common/
  goal_configs.py        mask_to_configuration (grid-then-jitter goal-as-
                        configuration) plus its legacy non-penetration bound
                        assert_no_penetration/_penetrates -- an axis-aligned
                        circumscribed-box test at the conservative any-yaw
                        diameter (MIN_PITCH = CUBE_SIZE*sqrt(2)). UNCHANGED
                        and still used by mask_to_configuration/goal
                        generation (claim C-021 depends on this exact
                        behaviour) -- do not migrate it. Measured: this bound
                        rejects legal diagonal contact by a factor of sqrt(2)
                        and rejects 91-99% of REAL DS-0002 states outright,
                        i.e. it cannot express "contact" and nothing could
                        ever be compacted against it -- see cube_overlap.py.
                        sample_synthetic_state (the old B1-scattered/B2-clump
                        no-target-mask generator) is RETIRED from DS-0003 as
                        of 2026-09-17 (superseded by pile_compaction.py) but
                        the function itself is untouched and still importable.
  cube_overlap.py        Exact separating-axis (SAT) overlap test for
                        yaw-rotated squares: overlaps_pairs, any_overlap_matrix,
                        state_is_legal(xy, yaw, size, tol). tol=0.0 is the
                        exact boundary; a NEGATIVE tol inflates the squares
                        (a safety margin used during placement so the
                        finished state still verifies at tol=0). This is the
                        legality test the new compaction generator relaxes
                        cubes against -- it replaces (for DS-0003 only)
                        goal_configs.py's conservative axis-aligned bound
                        above. Unit-tested on aligned/diagonal/45-degree/
                        contact cases.
  pile_compaction.py     sample_compacted_state(n_objects, rng, size,
                        compaction, n_sweeps, tol, seed_jitter, yaw_kappa) ->
                        (xy, yaw): grid seed (pitch sized from the ACTUAL yaw
                        spread, not the any-yaw worst case) then per-object
                        relaxation toward the centroid, each candidate move
                        checked against cube_overlap.py at a small negative
                        tol margin. `compaction` in [0,1] sets the effective
                        sweep count (the dispersed<->compact diversity axis);
                        `yaw_kappa` concentrates yaws around a random common
                        heading via von Mises (real cubes partially align and
                        pack tighter than uniform-random yaws allow -- do not
                        exceed yaw_kappa=6, the seed lattice starts too tight
                        above that and legality drops). Returns a CENTRED
                        pile; placing it in a workspace is the caller's job.
                        Backs datasets/DS-0003-synthetic-states/ (2026-09-17
                        rewrite, replacing sample_synthetic_state above).
  paired_stats.py       paired, state-resampled model comparison for every
                        benchmark: variance_components, friedman,
                        paired_comparison (bootstrap CI + sign-flip p + Holm),
                        rank_stability, required_n / power_table. The STATE is
                        the replication unit; input (models, states), higher =
                        better (EXP-0026)

Baselines/NFD/
  nfd_lib.py            the plain NFD baseline: registers dataset
                        `nfd-genesis-3ch` and model `nfd-unet3ch`
  predictor.py          plain (unwarped) `NFDPredictor`/`build_predictor[_2ch_ablation]`
                        eval-time predictor only -- the warped/residual/flow
                        predictor code that used to live here moved out to
                        `model/warped_nfd/`, `model/residual_nfd/`,
                        `model/flow_nfd/` (see the `model/` entries above):
                        those are new model families with their own
                        multi-file code, not minor parameter changes over
                        this baseline, so they belong under `model/`, not
                        `Baselines/`
  train_nfd.py          thin driver: imports `nfd_lib` plus
                        `model.warped_nfd.lib` / `model.residual_nfd.lib` /
                        `model.flow_nfd.lib` (registration side effects),
                        then runs `training.trainer.Trainer`
  configs/               training configs for every arm above (unmoved --
                        these are `Baselines/NFD/`-owned config instances,
                        not code)
```

Genesis-dependent modules (`env/genesis_env.py`, `simple_mpc/genesis_oracle.py`,
`Genesis/*`) are only importable where the `genesis` package is installed;
everything else in this map is Genesis-free. `simple_mpc/human_grid_search.py`
is a partial exception: `build_action_grid` is plain NumPy and Genesis-free,
but `grid_search_refine` needs a real `GenesisOracleEnv` — the `genesis`-
requiring import is deferred inside that one function so the module itself
stays importable without `genesis` installed (see docs/human_demo_design.md).


### Two adapter families in `simple_mpc/adapters.py`

`make_adapter` serves the live `run_simple_mpc` loop: observation-driven,
camera-aware, goal given as a distance-transform subgoal. It supports the
Eulerian wrapper and the GNN and raises `NotImplementedError` for everything
else, which is correct -- the other models are not observation-driven.

`make_occ_adapter` / `OCC_ADAPTERS` serve action optimisation against the
`Baselines/` occupancy models on the slate workspace. The split is deliberate
and is about the INPUT, not the model class: one family starts from a rendered
depth image and owns a camera convention; the other starts from particle
positions and owns the +/-64 mm / 64 px slate convention that
`scripts/probes/binned_pool_cache.py` and every DS-0001 number already use.
Merging them would force one of the two conventions onto the other.

The second family's reason to exist is that a `predict_occ` predictor is an
OFFLINE scorer (`@torch.no_grad()`): it can rank a fixed pool but cannot be
optimised against. `PredictorGradientAdapter` reuses the predictor's own
forward via `__wrapped__` rather than reimplementing it, so there is exactly
one copy of each model's prediction math.

## Data Flow

```
Training config YAML
  ├─ model:   configs/model/*.yaml      ──► build_model(cfg)   ──► ModelTrainingWrapper
  └─ dataset: configs/dataset/*.yaml    ──► build_dataset(cfg) ──► Dataset

Trainer.run()
  for batch in DataLoader:
      prediction = model_wrapper(batch)
      loss, comps = loss_fn(prediction, batch)
      metrics.update(prediction, batch)

  -> saves checkpoint + model_card.yaml
```

## Extension Points

### Add a Model

1. Implement an `nn.Module`.
2. Register a factory in `registry/model_registry.py`.
3. Add a config in `configs/model/`.
4. Reference it from a training config.

Example:

```python
@register_model("mymodel")
def _build_mymodel(cfg: dict) -> EulerianTrainingWrapper:
    from mypackage import MyModel
    model = MyModel(in_channels=cfg.get("in_channels", 2))
    return EulerianTrainingWrapper(model, uses_physics=False)
```

### Add a Dataset

1. Implement a builder in `registry/dataset_registry.py`.
2. Register it with `@register_dataset("name")`.
3. Ensure outputs match the standard batch contract in `INTERFACES.md`.
4. Add a dataset config in `configs/dataset/`.

### Add a Loss

1. Subclass `LossFn` in `training/losses.py`.
2. Register it with `@register_loss("name")`.
3. Reference it under `training.loss` in your training config.

## Config Structure

Top-level training config schema:

```yaml
model: configs/model/unetfilm.yaml      # must resolve to a dict with a `type` key
dataset: configs/dataset/genesis_cube.yaml

training:
  epochs: 100
  batch_size: 64
  lr: 1e-4
  lr_scheduler:            # optional StepLR params
    type: StepLR
    step_size: 100
    gamma: 0.75
  augmentation: true
  patience: 100            # early stopping
  mixed_precision: true
  grad_clip_norm: 1.0
  save_every_n_epochs: 10
  num_workers: 4
  loss:
    type: eulerian_combined   # default depends on model.type
                              # (lagrangian_mse for gnn-propnet)
    mse: 1.0
    mass: 0.2

# The inference block is NOT read during training: it is copied verbatim
# into model_card.yaml and consumed later by load_model_from_card
# (grid/plate geometry and raw physics values for normalization).
inference:
  representation: eulerian
  grid_n: 128

output:
  log_dir: runs/example
```

Notes:
- `training/trainer.py` resolves top-level string paths ending in `.yaml` and
  inlines them (project root first, then the config file's directory).
- The `model` dict must contain a `type` key matching a registered factory.
- `model_card.yaml` is written as a sidecar to best checkpoints.
- `pretrained_checkpoint` (top-level, optional) seeds training from an
  existing checkpoint (see `configs/training/unetfilm_finetune.yaml`).

## Running

```bash
python -m training.train configs/training/unetfilm_genesis.yaml
python -m training.train configs/training/unetfilm_genesis.yaml --eval-only
python -m training.train configs/training/unetfilm_genesis.yaml --override training.epochs=200
```

Oracle (Genesis-as-model) MPC — see `docs/oracle_mpc_design.md` for the design:

```bash
python -m Genesis.benchmark_n_envs                      # pick mpc.n_envs first
python run_oracle_mpc.py --config simple_mpc/config/config_oracle_test.yaml --save-video
```
