# Watch a full 60 s episode in a Pygame window (~60 s real time)
uv run python scripts/demo_episode.py

# Run the same episode headless at full speed
uv run python scripts/demo_episode.py --headless

# Report the score distribution of a random policy over 100 episodes (seed 0)
uv run python scripts/random_baseline.py

# Train a DQN agent for 200,000 steps (saved to models/dqn_gold_miner.zip)
uv run python scripts/train_dqn.py

# Evaluate the trained DQN agent for one headless episode
uv run python scripts/eval_dqn.py

# Replay the trained DQN agent in a Pygame window (~60 s real time)
uv run python scripts/eval_dqn.py --render
