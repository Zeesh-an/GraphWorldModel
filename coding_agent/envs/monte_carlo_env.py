"""
Ground-truth Monte-Carlo environment: drives the real NDlib simulator, applying
the strategy's actions at every timestep and averaging spread over mc_runs.

This is the baseline the world-model environment is compared against.
"""

import numpy as np

from coding_agent.types import ActionFn, GraphInfo, State, Trajectory
from coding_agent.tools.primitives import build_simulator


class MonteCarloEnvironment:
    def __init__(self, g: GraphInfo, diffusion_model: str, mc_runs: int = 30) -> None:
        self.g = g
        self.diffusion_model = diffusion_model
        self.mc_runs = mc_runs

    def rollout(
        self, action_fn: ActionFn, horizon: int, budget: int, seed: int = 0
    ) -> Trajectory:
        rng = np.random.default_rng(seed)
        final_counts: list[float] = []
        rep_states: list[State] = []
        rep_actions: list[list] = []
        rep_counts: list[float] = []

        for run in range(self.mc_runs):
            sim = build_simulator(
                self.g, self.diffusion_model, seed=int(rng.integers(1 << 30))
            )
            state = State(infected=[], frontier=[])
            states = [state]
            actions: list[list] = []
            counts = [0.0]

            for t in range(horizon + 1):
                bag = action_fn(state, t)
                state = sim.advance(bag)
                states.append(state)
                actions.append(bag)
                counts.append(float(len(state.infected)))
                if t > 0 and not state.frontier and not bag:
                    break
            final_counts.append(float(len(state.infected)))
            if run == 0:  # keep the first run as the representative trajectory
                rep_states, rep_actions, rep_counts = states, actions, counts

        reward = float(np.mean(final_counts)) if final_counts else 0.0
        rep_counts[-1] = reward  # report the averaged final count as the endpoint

        return Trajectory(
            states=rep_states,
            actions=rep_actions,
            reward=reward,
            infected_counts=rep_counts,
            cost={"mc_runs": self.mc_runs, "env": "monte_carlo"},
        )
