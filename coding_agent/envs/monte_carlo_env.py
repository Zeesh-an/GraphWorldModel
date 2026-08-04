"""
Ground-truth Monte-Carlo environment: drives the real NDlib simulator, applying the strategy's actions at every timestep and averaging spread over mc_runs.

This is the baseline the world-model environment is compared against.
"""

import time
import numpy as np

from coding_agent.types import ActionFn, GraphInfo, State, Trajectory, pad_counts
from coding_agent.tools.primitives import build_competitive_simulator, build_simulator
from data.wm_competitive import CompetitiveConfig
from data.wm_simulator import spent

seed_upper_bound = 1 << 30


class MonteCarloEnvironment:
    def __init__(
        self,
        graph: GraphInfo,
        diffusion_model: str,
        mc_runs: int = 30,
        base_seed: int = 0,
        remove_semantics: str = spent,
        negative_seeds: tuple = (),
        competitive_config: CompetitiveConfig | None = None,
    ) -> None:
        self.graph = graph
        self.diffusion_model = diffusion_model
        self.mc_runs = mc_runs
        self.remove_semantics = remove_semantics
        # Influence blocking: run the TWO-cascade simulator with S_N committed at
        # t=0 by reset(), and score the negative cascade. `reward` stays
        # `len(state.infected)` because State maps the negative cascade onto those
        # fields, so nothing downstream has to know which simulator ran.
        self.negative_seeds = tuple(int(node) for node in negative_seeds)
        self.competitive_config = competitive_config
        # Seed every rollout uses unless one is named explicitly. Shared across
        # candidates on purpose: common random numbers make the DIFFERENCE
        # between two strategies far better resolved than either absolute score.
        self.base_seed = base_seed
        # Cumulative real-environment episodes across all rollout calls — the
        # sample-efficiency axis for the native-agent condition (--mc-runs 1)
        self.episodes_used = 0
        # Cumulative inner-loop cost. evaluator_seconds is the axis the condition
        # ladder is read on: it grows with mc_runs x rounds here and stays flat in
        # WorldModelEnvironment, which is the whole claim. Reported per arm as
        # evaluator_calls / evaluator_seconds.
        self.rollout_calls = 0
        self.evaluator_seconds = 0.0

    def rollout(
        self, action_fn: ActionFn, horizon: int, budget: int, seed: int | None = None
    ) -> Trajectory:
        start = time.perf_counter()
        seed = self.base_seed if seed is None else seed
        rng = np.random.default_rng(seed)

        final_counts = []
        per_run_curves = []
        final_infected_freq = np.zeros(self.graph.num_nodes)
        representative_states = []
        representative_actions = []
        representative_counts = []

        self.episodes_used += self.mc_runs

        for run in range(self.mc_runs):
            # For each monte carlo run, build a fresh simulator with a child seed
            child_seed = int(rng.integers(seed_upper_bound))

            if self.competitive_config is not None:
                simulator = build_competitive_simulator(
                    self.graph,
                    self.diffusion_model,
                    self.negative_seeds,
                    seed=child_seed,
                    config=self.competitive_config,
                )
                # NOT empty: the rumour moved first and is already committed, which
                # is the premise of the whole task (§5.4, "first mover has a clear
                # advantage")
                state = simulator.current_state()
            else:
                simulator = build_simulator(
                    self.graph,
                    self.diffusion_model,
                    seed=child_seed,
                    remove_semantics=self.remove_semantics,
                )
                state = State(infected=[], frontier=[])

            states = [state]
            actions = []
            counts = [float(len(state.infected))]

            for timestep in range(horizon + 1):
                # Each timestep, get the action bag and apply the action affect (T_exo) and one diffusion step (T_endo)
                bag = action_fn(state, timestep)
                state = simulator.advance(bag)

                states.append(state)
                actions.append(bag)
                counts.append(float(len(state.infected)))

                # Terminate early once the cascade is dead (empty frontier) and the
                # strategy is idle. Under competition BOTH cascades have to be dead:
                # a live counter-cascade is still changing which nodes are protected.
                alive = state.frontier or state.pos_frontier
                if timestep > 0 and not alive and not bag:
                    break

            final_counts.append(float(len(state.infected)))
            per_run_curves.append(pad_counts(counts, horizon))
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

        elapsed = time.perf_counter() - start
        self.rollout_calls += 1
        self.evaluator_seconds += elapsed

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
                "rollout_seconds": elapsed,
            },
            final_marginals=(final_infected_freq / self.mc_runs).round(3).tolist(),
            spread_curve=np.mean(per_run_curves, axis=0).round(4).tolist(),
        )
