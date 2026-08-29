# gold-miner-sim

Minimal, deterministic Gold Miner Gymnasium environment
(`src/gold_miner_sim/`). One `step()` advances the simulation by exactly
one physics tick (1/60 s).

## Map modes

- `GoldMinerEnv(map_mode="fixed")` — the default and the V0 contract:
  GOLD/DIAMOND/ROCK always spawn at the same fixed centers.
  `scripts/demo_episode.py` still uses the fixed map.
- `GoldMinerEnv(map_mode="random")` — every `reset(seed=s)` draws 3 of
  the 12 `RANDOM_SPAWN_POINTS` without replacement, one each for
  GOLD/DIAMOND/ROCK. Only the initial positions vary; object properties,
  the observation contract, physics and determinism are unchanged, and a
  given seed always reproduces the same map.

Training uses the random map; the demo stays on the fixed map. The current
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

- Milestones 2-3 trained and evaluated on `DecisionIntervalWrapper`
  (`src/gold_miner_sim/wrappers.py`): every agent action is repeated for
  a fixed 10 physics ticks per `step()`. It is kept as a baseline.
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
separate aggregate reported by `scripts/run_ablation.py`.

## Commands

Training-related scripts need the `train` dependency group
(`uv run --group train ...`).

```bash
# Watch a full 60 s fixed-map episode in a Pygame window (~60 s real time)
uv run python scripts/demo_episode.py

# Run the same episode headless at full speed
uv run python scripts/demo_episode.py --headless

# M6 random-policy baseline over the benchmark maps (seeds 1000-1099)
uv run --group train python scripts/random_baseline.py --episodes 100 --seed 1000

# Train the M6 DQN for 200,000 steps on random maps with a 3-FIRE budget
# -> models/dqn_gold_miner_fire_budget.zip + Monitor logs in runs/dqn_fire_budget/
uv run --group train python scripts/train_dqn.py --timesteps 200000 --seed 0

# Evaluate the trained agent headless over the same 100 maps
uv run --group train python scripts/eval_dqn.py --model models/dqn_gold_miner_fire_budget.zip --episodes 100 --seed 1000

# Replay the trained agent in a Pygame window on individual benchmark maps
uv run --group train python scripts/eval_dqn.py --model models/dqn_gold_miner_fire_budget.zip --episodes 1 --seed 1000 --render
uv run --group train python scripts/eval_dqn.py --model models/dqn_gold_miner_fire_budget.zip --episodes 1 --seed 1007 --render
uv run --group train python scripts/eval_dqn.py --model models/dqn_gold_miner_fire_budget.zip --episodes 1 --seed 1042 --render

# M7 observation ablation: train one Full + one Blind model for a single seed
# -> models/ablation/<mode>/seed_<n>.zip + Monitor logs in runs/ablation/<mode>/seed_<n>/
uv run --group train python scripts/train_dqn.py --observation full --timesteps 200000 --seed 0 --output models/ablation/full/seed_0.zip
uv run --group train python scripts/train_dqn.py --observation blind --timesteps 200000 --seed 0 --output models/ablation/blind/seed_0.zip

# Run the full paired experiment (10 sequential 200k runs, seeds 0-4) and
# write the paired summary to runs/ablation/results.json
uv run --group train python scripts/run_ablation.py

# Evaluate one ablation model on the benchmark maps (match the training mode!)
uv run --group train python scripts/eval_dqn.py --model models/ablation/blind/seed_0.zip --observation blind --episodes 100 --seed 1000 --json-output runs/ablation/blind/seed_0/eval.json

# Compare FIRE angles of a paired Full/Blind model on selected maps (headless)
uv run --group train python scripts/replay_ablation.py --full-model models/ablation/full/seed_0.zip --blind-model models/ablation/blind/seed_0.zip
```

The evaluation output includes `mean_random`, `mean_dqn`, and
`delta = mean_dqn - mean_random`; the DQN metrics are computed on exactly the
same map seeds as the random baseline. The `--render` commands above are
single-map replays and retain the human Pygame view without changing the
headless 100-map benchmark.
