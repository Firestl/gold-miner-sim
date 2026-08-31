# gold-miner-sim

Minimal, deterministic Gold Miner Gymnasium environment
(`src/gold_miner_sim/`). One `step()` advances the simulation by exactly
one physics tick (1/60 s).

## Map modes

- `GoldMinerEnv(map_mode="fixed")` — the default and the V0 contract:
  GOLD/DIAMOND/ROCK always spawn at the same fixed centers.
- `GoldMinerEnv(map_mode="random")` — every `reset(seed=s)` draws 3 of
  the 12 `RANDOM_SPAWN_POINTS` without replacement, one each for
  GOLD/DIAMOND/ROCK. Only the initial positions vary; object properties,
  the observation contract, physics and determinism are unchanged, and a
  given seed always reproduces the same map.

Training uses the random map. The current
Milestone 6 benchmark adds a three-FIRE episode budget outside the physics
environment, so the policy receives a 27th observation value containing the
normalized FIREs remaining.

## Two different `--seed` meanings

- `scripts/train_dqn.py --seed` seeds both the DQN experiment (model,
  exploration, replay buffer) and the environment RNG, so the training
  map sequence is fully determined by this seed. This matches Issue #5 §12:
  "seeds SB3 / environment reproducibility."
- `scripts/random_baseline.py --seed` and `scripts/eval_dqn.py --seed`
  are **evaluation map seed start values**: episode `i` (0-based) plays
  the map drawn from `seed + i`.

Baseline and evaluation therefore default to the same map set, seeds
1000–1099 — that range is the test set, and any model is only comparable
across runs if it is scored on exactly these maps. Both scripts report
`mean`, `std`, `min`, `max`, `full_score_count` (score exactly 800), and
`full_score_rate`.

## Decision timing

- Milestones 2-3 trained and evaluated on `DecisionIntervalWrapper`:
  every agent action was repeated for a fixed 10 physics ticks per
  `step()`.
- Milestone 4 used `SwingDecisionWrapper`: the agent only decides while
  the hook is SWINGING. `WAIT` advances the simulation by up to 10 ticks;
  `FIRE` automatically plays out the whole extend/retract round trip and
  puts all accumulated reward into that single transition (transitions
  are variable-length by design). It exposed an angle-pinning loop: after
  a FIRE the hook returns to the exact firing angle, so the next decision
  observation is identical to the previous one.
- From Milestone 5 on, training and evaluation use
  `SwingAdvanceDecisionWrapper`, which keeps those semantics but, after a
  completed FIRE cycle, swings on for another 10 WAIT ticks before
  returning. The next decision is therefore never taken at the original
  firing angle.
- Milestone 6 wraps that chain in `FireBudgetWrapper(max_fires=3)`. WAIT does
  not consume budget; every FIRE consumes one budget even when the hook is
  empty or times out. The third FIRE transition completes normally (including
  its reward and post-FIRE advance), then ends the episode if the inner
  environment has not already timed out.
- Milestone 7 keeps that chain unchanged and adds
  `ObjectPositionMaskWrapper` as the "blind" condition of the observation
  ablation: it zeroes only the six object position slots (indices
  8, 9, 14, 15, 20, 21 = GOLD/DIAMOND/ROCK x, y) of the 27-dim observation.
  Everything else — hook state, radii, values, retract speeds, active flags,
  FIRE budget — stays visible, and rewards/flags/info pass through untouched.

Random-baseline numbers from earlier wrapper setups are not directly
comparable and need to be re-measured per wrapper.

## Observation ablation benchmark (Milestone 7)

Issue #13 measures whether the M6 DQN actually uses object position
information: under identical benchmark, DQN config and 200k-step budget,
only the object `(x, y)` observation slots differ between two conditions
trained and evaluated separately (no inference-time masking):

- **Full**: the complete 27-dim observation.
- **Blind**: positions masked to 0 by `ObjectPositionMaskWrapper` (applied
  outside `FireBudgetWrapper` via
  `gold_miner_sim.benchmark.make_benchmark_env("blind")`).

Each condition is trained with 5 paired training seeds (0–4, 10 runs total),
evaluated deterministically on the same 100 benchmark maps (seeds 1000–1099),
and compared per seed as `paired_delta = full_mean - blind_mean`. `std` in
the per-model evaluation output is the across-map episode std
(`std_episode` in the JSON output); the across-training-seed std is a
separate aggregate computed from the paired per-seed results.

## Geometry Oracle baseline (Milestone 8)

`scripts/oracle_baseline.py` is a deterministic, observation-only policy. It
uses the public hook geometry and the Full benchmark observation to predict
the first object intersected by the current legal ray, with target priority
GOLD → DIAMOND → ROCK. It does not train a model or read per-episode state
from the environment.

## Strong DQN v2 (Milestone 10)

`scripts/train_strong_dqn.py` is an independent training entry point. Its
default recipe uses `gamma=1.0`, `n_steps=5`, a `[256, 256]` MLP, a 200k
replay buffer, batch size 256, and a 1M-transition budget. Every 25k
transitions it evaluates deterministic behavior on maps 1000–1019 and saves
the highest-mean checkpoint to
`models/strong_dqn/seed_<seed>/best_model.zip`; the final learner is saved
separately as `final_model.zip`. The per-seed history is written to
`runs/strong_dqn/seed_<seed>/evaluations.json`.

Run the frozen recipe sequentially for seeds 0–4 with
`scripts/run_strong_dqn.py`. After the recipe and best checkpoints are fixed,
`scripts/eval_strong_dqn.py --split validation` evaluates maps 1000–1099 and
`scripts/eval_strong_dqn.py --split test` evaluates the held-out maps
2000–2099.

## Geometry generalization (Milestone 11)

Issue #21 freezes the five M10 best checkpoints and only changes the
evaluation spawn pool: `id` (the unchanged `RANDOM_SPAWN_POINTS` control),
`rot3` (the whole pool rotated +3 degrees around the anchor) and
`rot3_scale105` (+3 degrees and radial distance ×1.05). Maps 3000–3099 are
the paired evaluation seeds; the same map seed selects the same three pool
indices under every mode.

1. Oracle solvability gate — the M8 oracle must score mean 800, 100% full
   score and 0 miss on all three modes before any DQN OOD score is
   interpreted (`--geometry id` keeps the historical benchmark path
   unchanged):

```bash
uv run python scripts/oracle_baseline.py --episodes 100 --seed 3000 --geometry id
uv run python scripts/oracle_baseline.py --episodes 100 --seed 3000 --geometry rot3
uv run python scripts/oracle_baseline.py --episodes 100 --seed 3000 --geometry rot3_scale105
```

2. Frozen DQN geometry evaluation — 5 training seeds × 3 modes × 100 maps,
   `predict(obs, deterministic=True)` only, results in
   `runs/geometry_generalization/results.json` (with per-seed drop/retention
   tables and per-map paired deltas):

```bash
uv run --group train python scripts/eval_geometry_generalization.py
```

3. Behavior replay — per-FIRE trace lines (current angle, active center
   angles, predicted first hit, collected slot, score after fire) instead of
   the full evaluation:

```bash
uv run --group train python scripts/eval_geometry_generalization.py --trace --trace-seeds 0,3 --trace-maps 3000,3007,3042
```

Checkpoint freeze contract: the evaluator checks every requested
`models/strong_dqn/seed_<n>/best_model.zip` before starting and exits with
a non-zero status listing all missing files. It never retrains, never
fine-tunes and never skips a missing checkpoint. Each FIRE transition is
classified as `productive` (an object active flag went 1→0), `timeout_fire`
(no change and the transition ended by timeout) or `miss` (no change,
episode continued); timeouts are never counted as plain misses.

## Commands

Training-related scripts need the `train` dependency group
(`uv run --group train ...`).

```bash
# M6 random-policy baseline over the benchmark maps (seeds 1000-1099)
uv run --group train python scripts/random_baseline.py --episodes 100 --seed 1000

# Train the M6 DQN for 200,000 steps on random maps with a 3-FIRE budget
# -> models/dqn_gold_miner_fire_budget.zip + Monitor logs in runs/dqn_gold_miner_fire_budget/
# (without --output the model would land on the M7 default
# models/ablation/<observation>/seed_<seed>.zip instead)
uv run --group train python scripts/train_dqn.py --timesteps 200000 --seed 0 --output models/dqn_gold_miner_fire_budget.zip

# Evaluate the trained agent headless over the same 100 maps
uv run --group train python scripts/eval_dqn.py --model models/dqn_gold_miner_fire_budget.zip --episodes 100 --seed 1000

# Replay the trained agent in a Pygame window on individual benchmark maps
uv run --group train python scripts/eval_dqn.py --model models/dqn_gold_miner_fire_budget.zip --episodes 1 --seed 1000 --render
uv run --group train python scripts/eval_dqn.py --model models/dqn_gold_miner_fire_budget.zip --episodes 1 --seed 1007 --render
uv run --group train python scripts/eval_dqn.py --model models/dqn_gold_miner_fire_budget.zip --episodes 1 --seed 1042 --render

# M7 observation ablation: train one Full + one Blind model for a single seed.
# Omitting --output writes models/ablation/<observation>/seed_<seed>.zip
# (plus the derived Monitor dir runs/ablation/<observation>/seed_<seed>/),
# so modes and seeds never collide and M6 artifacts are never overwritten.
# -> models/ablation/<mode>/seed_<n>.zip + Monitor logs in runs/ablation/<mode>/seed_<n>/
uv run --group train python scripts/train_dqn.py --observation full --timesteps 200000 --seed 0
uv run --group train python scripts/train_dqn.py --observation blind --timesteps 200000 --seed 0

# Evaluate one ablation model on the benchmark maps (match the training mode!
# --model is required with --observation blind)
uv run --group train python scripts/eval_dqn.py --model models/ablation/blind/seed_0.zip --observation blind --episodes 100 --seed 1000 --json-output runs/ablation/blind/seed_0/eval.json

# M8 observation-only Geometry Oracle over the benchmark maps (seeds 1000-1099)
uv run python scripts/oracle_baseline.py --episodes 100 --seed 1000

# M10 Strong DQN v2: one seed (run on the GPU server)
uv run --group train python scripts/train_strong_dqn.py --seed 0

# M10 Strong DQN v2: sequential seeds 0-4 and 100-map validation
uv run --group train python scripts/run_strong_dqn.py

# M10 held-out evaluation, only after freezing the recipe/checkpoints
uv run --group train python scripts/eval_strong_dqn.py --split test

# Trace FIRE decisions for a selected map, or open the human renderer
uv run python scripts/oracle_baseline.py --episodes 1 --seed 1000 --trace
uv run python scripts/oracle_baseline.py --episodes 1 --seed 1000 --render
```

The evaluation output includes `mean_random`, `mean_dqn`, and
`delta = mean_dqn - mean_random`; the DQN metrics are computed on exactly the
same map seeds as the random baseline. The `--render` commands above are
single-map replays and retain the human Pygame view without changing the
headless 100-map benchmark.
