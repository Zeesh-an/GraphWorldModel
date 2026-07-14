"""
Ground-truth Monte-Carlo environment: drives the real NDlib simulator, applying
the strategy's actions at every timestep and averaging spread over mc_runs.

This is the baseline the world-model environment is compared against.
"""

import numpy as np

from coding_agent.types import ActionFn, GraphInfo, State, Trajectory
from coding_agent.tools.primitives import build_simulator

seed_upper_bound = 1 << 30


class MonteCarloEnvironment:
    def __init__(
        self, graph: GraphInfo, diffusion_model: str, mc_runs: int = 30
    ) -> None:
        self.graph = graph
        self.diffusion_model = diffusion_model
        self.mc_runs = mc_runs

    def rollout(
        self, action_fn: ActionFn, horizon: int, budget: int, seed: int = 0
    ) -> Trajectory:
        rng = np.random.default_rng(seed)
        final_counts = []
        representative_states = []
        representative_actions = []
        representative_counts = []

        for run in range(self.mc_runs):
            simulator = build_simulator(
                self.graph,
                self.diffusion_model,
                seed=int(rng.integers(seed_upper_bound)),
            )
            state = State(infected=[], frontier=[])
            states = [state]
            actions = []
            counts = [0.0]

            for timestep in range(horizon + 1):
                bag = action_fn(state, timestep)
                state = simulator.advance(bag)
                states.append(state)
                actions.append(bag)
                counts.append(float(len(state.infected)))
                if timestep > 0 and not state.frontier and not bag:
                    break

            final_counts.append(float(len(state.infected)))
            if run == 0:  # keep the first run as the representative trajectory
                representative_states = states
                representative_actions = actions
                representative_counts = counts

        reward = float(np.mean(final_counts)) if final_counts else 0.0

        # Report the averaged final count as the endpoint.
        representative_counts[-1] = reward

        return Trajectory(
            states=representative_states,
            actions=representative_actions,
            reward=reward,
            infected_counts=representative_counts,
            cost={"mc_runs": self.mc_runs, "env": "monte_carlo"},
        )
