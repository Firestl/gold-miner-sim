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

Training uses the random map; the demo stays on the fixed map.

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
across runs if it is scored on exactly these maps.

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

Random-baseline numbers from earlier wrapper setups are not directly
comparable and need to be re-measured per wrapper.

## Commands

Training-related scripts need the `train` dependency group
(`uv run --group train ...`).

```bash
# Watch a full 60 s fixed-map episode in a Pygame window (~60 s real time)
uv run python scripts/demo_episode.py

# Run the same episode headless at full speed
uv run python scripts/demo_episode.py --headless

# Random-policy baseline over 100 random maps (map seeds 1000-1099)
uv run --group train python scripts/random_baseline.py --episodes 100 --seed 1000

# Train DQN for 200,000 steps on random maps
# -> models/dqn_gold_miner_advance.zip + Monitor logs in runs/dqn_advance/
uv run --group train python scripts/train_dqn.py --timesteps 200000 --seed 0

# Evaluate the trained agent headless over 100 random maps (map seeds 1000-1099)
uv run --group train python scripts/eval_dqn.py --model models/dqn_gold_miner_advance.zip --episodes 100 --seed 1000

# Replay the trained agent in a Pygame window on the map from seed 1007 (~60 s real time)
uv run --group train python scripts/eval_dqn.py --model models/dqn_gold_miner_advance.zip --seed 1007 --render
```
