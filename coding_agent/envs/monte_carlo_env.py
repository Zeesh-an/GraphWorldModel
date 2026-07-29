"""
Ground-truth Monte-Carlo environment: drives the real NDlib simulator, applying the strategy's actions at every timestep and averaging spread over mc_runs.

This is the baseline the world-model environment is compared against.
"""

import time
import numpy as np

from coding_agent.types import ActionFn, GraphInfo, State, Trajectory
from coding_agent.tools.primitives import build_simulator

seed_upper_bound = 1 << 30


class MonteCarloEnvironment:
    def __init__(
        self,
        graph: GraphInfo,
        diffusion_model: str,
        mc_runs: int = 30,
        base_seed: int = 0,
    ) -> None:
        self.graph = graph
        self.diffusion_model = diffusion_model
        self.mc_runs = mc_runs
        # Seed every rollout uses unless one is named explicitly. Shared across
        # candidates on purpose: common random numbers make the DIFFERENCE
        # between two strategies far better resolved than either absolute score.
        self.base_seed = base_seed
        # Cumulative real-environment episodes across all rollout calls — the
        # sample-efficiency axis for the native-agent condition (--mc-runs 1)
        self.episodes_used = 0

    def rollout(
        self, action_fn: ActionFn, horizon: int, budget: int, seed: int | None = None
    ) -> Trajectory:
        start = time.perf_counter()
        seed = self.base_seed if seed is None else seed
        rng = np.random.default_rng(seed)

        final_counts = []
        final_infected_freq = np.zeros(self.graph.num_nodes)
        representative_states = []
        representative_actions = []
        representative_counts = []

        self.episodes_used += self.mc_runs

        for run in range(self.mc_runs):
            # For each monte carlo run, build a fresh NDlib simulator with a child seed
            simulator = build_simulator(
                self.graph,
                self.diffusion_model,
                seed=int(rng.integers(seed_upper_bound)),
            )

            # Start from an empty state
            state = State(infected=[], frontier=[])
            states = [state]
            actions = []
            counts = [0.0]

            for timestep in range(horizon + 1):
                # Each timestep, get the action bag and apply the action affect (T_exo) and one diffusion step (T_endo)
                bag = action_fn(state, timestep)
                state = simulator.advance(bag)

                states.append(state)
                actions.append(bag)
                counts.append(float(len(state.infected)))

                # Terminate early once the cascade is dead (empty frontier) and the strategy is idle (empty bag)
                if timestep > 0 and not state.frontier and not bag:
                    break

            final_counts.append(float(len(state.infected)))
            final_infected_freq[list(state.infected)] += 1.0

            # Keep the first run as the representative trajectory
            if run == 0:
                representative_states = states
                representative_actions = actions
                representative_counts = counts

        # The reward is the mean of the final infected node counts
        reward = float(np.mean(final_counts)) if final_counts else 0.0

        reward_se = (
            float(np.std(final_counts, ddof=1) / np.sqrt(len(final_counts)))
            if len(final_counts) > 1
            else 0.0
        )

        # Report the averaged final count as the endpoint
        representative_counts[-1] = reward

        return Trajectory(
            states=representative_states,
            actions=representative_actions,
            reward=reward,
            infected_counts=representative_counts,
            cost={
                "mc_runs": self.mc_runs,
                "env": "monte_carlo",
                # The seed this rollout ran under: every per-episode simulator
                # seed is drawn from it, so replaying it reproduces the number
                "seed": int(seed),
                "reward_se": reward_se,
                "rollout_seconds": time.perf_counter() - start,
            },
            final_marginals=(final_infected_freq / self.mc_runs).round(3).tolist(),
        )
