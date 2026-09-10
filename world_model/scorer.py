"""
`WorldModelScorer` — the stable surface the coding agent loads a checkpoint
through.

The contract this exists to establish: the coding-agent side should never have to
know which files live in `world_model/`, which head a checkpoint used, or how a
feature matrix is laid out. It loads a checkpoint and asks "which of these
candidates is better".

Everything here delegates. Rollouts run through `WorldModelEnvironment`, which is
the tested implementation of the ensemble loop (coupled sampling, block-diagonal
batching, copy-on-write edge state); reconstruction runs through
`world_model.checkpoint.build_model`. This class adds an API, not a second
implementation — a scorer that reimplemented the rollout would be a second thing
to keep in sync with the evaluation suite, and the two would diverge on the first
bug fix.

Offline use only. §0 of the brief is explicit that the world model ranks
candidate algorithms during design and is not packaged into the algorithm that
ships; nothing here is imported by generated strategy code.
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from world_model.checkpoint import ModelSpec, describe, load_checkpoint
from world_model.wm_data import build_features, build_graph_input, edges_to_arrays


@dataclass(frozen=True)
class ScoringContext:
    """
    Everything a candidate is scored UNDER, held apart from the candidate itself.

    Separate object because two candidates are only comparable when they were
    scored under the same horizon, budget, ensemble size and seed. Passing these
    per-call invites a comparison between a 20-sample estimate and a 5-sample
    one, which is a real risk when the caller is a refinement loop that tunes its
    own cost.
    """

    horizon: int = 20
    budget: int = 5
    n_samples: int = 20
    seed: int = 0

    @classmethod
    def coerce(cls, context) -> "ScoringContext":
        if context is None:
            return cls()

        if isinstance(context, ScoringContext):
            return context

        return cls(**{key: value for key, value in dict(context).items()
                      if key in {"horizon", "budget", "n_samples", "seed"}})


@dataclass(frozen=True)
class CandidateScore:
    """One candidate's predicted quality, with enough context to be auditable."""

    index: int
    name: str
    score: float
    #: E|infected| after each timestep, ensemble-mean. The scalar `score` is its
    #: endpoint; the curve is what a feedback prompt can actually reason about.
    spread_curve: list[float] | None = None
    error: str | None = None

    @property
    def valid(self) -> bool:
        return self.error is None


class WorldModelScorer:
    """
    Load once, score many.

        scorer = WorldModelScorer.load("run/world_model/wm_sage_IC.pt")
        ranked = scorer.rank_candidates(graph, candidates, context)

    `graph` is a `coding_agent.types.GraphInfo`; a candidate is either an
    `ActionFn` — `(State, timestep) -> list[ActionOp]`, the same contract every
    environment in this repository consumes — or a plan, `list[list[ActionOp]]`,
    which is what the one-shot and evolve methods emit.
    """

    def __init__(
        self,
        model: nn.Module,
        spec: ModelSpec,
        device: str = "cpu",
        train_meta: dict | None = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.spec = spec
        self.device = device
        self.train_meta = dict(train_meta or {})

    # -- construction --------------------------------------------------------

    @classmethod
    def load(
        cls,
        checkpoint: str | Path,
        config: str | Path | dict | ModelSpec | None = None,
        device: str = "cpu",
        strict_spec: bool = True,
    ) -> "WorldModelScorer":
        """
        Rebuild a scorer from a checkpoint.

        `config` is needed only for a legacy bare-state_dict checkpoint, which
        does not carry its own architecture; `load_checkpoint` raises
        `LegacyCheckpointError` naming the file to pass. For a self-describing
        checkpoint it is optional, and a disagreement is an error rather than a
        silent preference.
        """
        model, spec, train_meta = load_checkpoint(
            checkpoint, config=config, device=device, strict_spec=strict_spec
        )

        return cls(model, spec, device=device, train_meta=train_meta)

    @classmethod
    def from_results_json(
        cls, results_json: str | Path, device: str = "cpu"
    ) -> "WorldModelScorer":
        """
        Load via a `train_wm.py` results JSON, deriving the checkpoint path from
        its config — the layout `train_wm` has always written.

        Kept because every checkpoint produced before this module existed is
        reachable only this way.
        """
        import json

        config = json.loads(Path(results_json).read_text())["config"]
        checkpoint = (
            Path(config["ckpt_dir"])
            / f"wm_{config['model']}_{config['diffusion_model']}.pt"
        )

        return cls.load(checkpoint, config=results_json, device=device)

    def describe(self) -> str:
        return (
            f"{self.spec.backbone}/{self.spec.head} {self.spec.diffusion_model} "
            f"remove={self.spec.remove_semantics} "
            f"encoding={self.spec.action_encoding} "
            f"hide_w={self.spec.hide_edge_weights}"
        )

    # -- one-step prediction -------------------------------------------------

    @torch.inference_mode()
    def predict_transition(self, graph, state, action) -> dict[str, np.ndarray]:
        """
        One step of `f_theta`: `(G, s_t, a_t) -> P(infected), P(frontier)`.

        Returns per-node marginals, not a sampled state. The caller decides
        whether to threshold or sample; the evaluation suite and the rollout make
        opposite choices there, and baking one in would silently pick for them.

        Edge ops in `action` are applied to the adjacency before the forward
        pass, matching what the data generator recorded: for IC and LT the
        post-action graph is the sufficient dynamical object.
        """
        from data.wm_simulator import ActionOp
        from world_model.wm_data import apply_edge_ops

        bag = [op.to_dict() if isinstance(op, ActionOp) else dict(op) for op in action]
        base_edges = {
            (int(graph.edge_index[0, edge]), int(graph.edge_index[1, edge])): float(
                graph.ic_probs[edge]
            )
            for edge in range(graph.edge_index.shape[1])
        }
        edge_index, weights = edges_to_arrays(apply_edge_ops(base_edges, bag))

        record = {
            "state": {
                "infected": sorted(int(node) for node in state.infected),
                "frontier": sorted(int(node) for node in state.frontier),
            },
            "action": bag,
            "next_state": {"infected": [], "frontier": []},
            # Targets are unread here — only X is built — but build_features
            # requires the keys, so a caller cannot accidentally score against a
            # dataset row that never had marginals.
            "next_marginal_infected": {},
            "next_marginal_frontier": {},
        }
        features, _, _ = build_features(
            record, edge_index, graph.num_nodes, self.spec.action_encoding
        )
        graph_input = build_graph_input(
            edge_index,
            weights,
            graph.num_nodes,
            self.spec.diffusion_model,
            torch.device(self.device),
            self.spec.hide_edge_weights,
        )
        probabilities = (
            torch.sigmoid(
                self.model(
                    torch.from_numpy(features).to(self.device), graph_input
                )
            )
            .cpu()
            .numpy()
        )

        return {
            "infected": probabilities[:, 0],
            "frontier": probabilities[:, 1],
        }

    @torch.inference_mode()
    def edge_transmission(self, graph, state, action) -> dict[str, np.ndarray]:
        """
        The learned per-arc transmission `q(u -> v)`, beside the true `w(u -> v)`.

        The EDGE-level marginal that `predict_transition` aggregates away. Under
        the IC structured head `q` is the entire learned content of the model —
        everything else in the transition is hand-written — so this is the one
        view that shows what the weights actually fit, and it is what a
        bottleneck diagnostic over ARCS (rather than nodes) needs.

        Returns `edge_index` (2, E), `q` (E,) and `w` (E,) so a caller can join
        them; `q_minus_w` is the residual the learning is responsible for. IC
        heads only — the LT head carries no per-arc transmission at all, and
        raises rather than returning a number that does not mean this.
        """
        from data.wm_simulator import ActionOp
        from world_model.wm_data import apply_edge_ops

        head = getattr(self.model, "head", None)

        if not hasattr(head, "transmission"):
            raise ValueError(
                f"head {type(head).__name__} carries no per-arc transmission; "
                f"only the IC structured heads do"
            )

        bag = [op.to_dict() if isinstance(op, ActionOp) else dict(op) for op in action]
        base_edges = {
            (int(graph.edge_index[0, edge]), int(graph.edge_index[1, edge])): float(
                graph.ic_probs[edge]
            )
            for edge in range(graph.edge_index.shape[1])
        }
        edge_index, weights = edges_to_arrays(apply_edge_ops(base_edges, bag))

        record = {
            "state": {
                "infected": sorted(int(node) for node in state.infected),
                "frontier": sorted(int(node) for node in state.frontier),
            },
            "action": bag,
            "next_state": {"infected": [], "frontier": []},
            "next_marginal_infected": {},
            "next_marginal_frontier": {},
        }
        features, _, _ = build_features(
            record, edge_index, graph.num_nodes, self.spec.action_encoding
        )
        graph_input = build_graph_input(
            edge_index,
            weights,
            graph.num_nodes,
            self.spec.diffusion_model,
            torch.device(self.device),
            self.spec.hide_edge_weights,
        )
        hidden = self.model.encoder(
            torch.from_numpy(features).to(self.device), graph_input
        )
        q = head.transmission(hidden, graph_input).cpu().numpy()

        return {
            "edge_index": edge_index,
            "q": q,
            "w": weights,
            "q_minus_w": q - weights,
        }

    # -- rollout -------------------------------------------------------------

    def environment(self, graph, context: ScoringContext | None = None):
        """
        A `WorldModelEnvironment` bound to this checkpoint and graph.

        Imported lazily: `coding_agent.envs` imports `world_model`, so a
        module-level import here would close the cycle. Exposed rather than kept
        private because the refinement loop wants the environment's cost counters
        (`forward_passes`, `rollout_calls`), which are what a trusted-call
        reduction is measured against.
        """
        from coding_agent.envs.world_model_env import WorldModelEnvironment

        context = ScoringContext.coerce(context)

        return WorldModelEnvironment(
            self.model,
            graph,
            self.spec.diffusion_model,
            device=self.device,
            n_samples=context.n_samples,
            base_seed=context.seed,
            remove_semantics=self.spec.remove_semantics,
            hide_edge_weights=self.spec.hide_edge_weights,
            action_encoding=self.spec.action_encoding,
        )

    def diagnostics(
        self,
        graph,
        context: ScoringContext | None = None,
        region_scheme: str = "community",
        paired: bool = True,
        sense: str = "maximize",
    ):
        """
        A `PlanDiagnostics` bound to this checkpoint and graph.

            wm = WorldModelScorer.load(ckpt).diagnostics(graph, context)
            wm.evaluate(plan); wm.probe_drop(plan, v); wm.explain_plan(plan)

        The probe surface of research §Part 4, reached from a checkpoint rather
        than from an already-built environment, which is what an offline analysis
        script has. Imported lazily for the same reason `environment` is: the
        `coding_agent` package imports `world_model`.
        """
        from coding_agent.diagnostics import PlanDiagnostics

        context = ScoringContext.coerce(context)

        return PlanDiagnostics(
            self.environment(graph, context),
            graph,
            horizon=context.horizon,
            budget=context.budget,
            seed=context.seed,
            region_scheme=region_scheme,
            paired=paired,
            sense=sense,
        )

    def rollout(self, graph, candidate, context: ScoringContext | None = None):
        """
        Roll a candidate forward under the model and return a `Trajectory`.

        Starts from the empty state, which is what `WorldModelEnvironment.rollout`
        implements and what every task in `pipeline.tasks` currently needs — the
        planner seeds the cascade for a maximize task, and a containment task's
        outbreak is injected by the strategy's own `t=0` bag. Rolling from an
        arbitrary mid-cascade state is a different entry point into the ensemble
        loop and is deliberately not faked here.
        """
        context = ScoringContext.coerce(context)
        environment = self.environment(graph, context)

        return environment.rollout(
            as_action_fn(candidate),
            horizon=context.horizon,
            budget=context.budget,
            seed=context.seed,
        )

    # -- candidate scoring ---------------------------------------------------

    def score_candidate(self, graph, candidate, context=None) -> float:
        """Predicted final spread. Higher is better for a maximize task."""
        return float(self.rollout(graph, candidate, context).reward)

    def score_candidates(self, graph, candidates, context=None) -> list[CandidateScore]:
        """
        Score every candidate under ONE context, preserving input order.

        A candidate that raises is recorded as invalid rather than dropped or
        allowed to abort the batch: generated code fails routinely, and a batch
        that silently shrinks turns "the model ranked 12 candidates" into a claim
        about however many happened to run.
        """
        context = ScoringContext.coerce(context)
        scores = []

        for index, candidate in enumerate(candidates):
            name = candidate_name(candidate, index)

            try:
                trajectory = self.rollout(graph, candidate, context)
                scores.append(
                    CandidateScore(
                        index=index,
                        name=name,
                        score=float(trajectory.reward),
                        spread_curve=trajectory.spread_curve,
                    )
                )
            except Exception as exception:  # noqa: BLE001 - recorded, not raised
                scores.append(
                    CandidateScore(
                        index=index,
                        name=name,
                        score=float("-inf"),
                        error=f"{type(exception).__name__}: {exception}",
                    )
                )

        return scores

    def rank_candidates(
        self, graph, candidates, context=None, sense: str = "maximize"
    ) -> list[CandidateScore]:
        """
        Score, then order best-first under `sense`.

        `sense` rather than a hardcoded descending sort because a containment task
        minimizes spread; `pipeline.conditions.result_sense` is the caller's
        source for it. Invalid candidates sort last in both senses — they carry no
        information about ordering and must never win a verification slot.
        """
        if sense not in ("maximize", "minimize"):
            raise ValueError(f"sense must be maximize or minimize, got {sense!r}")

        scores = self.score_candidates(graph, candidates, context)
        reverse = sense == "maximize"

        return sorted(
            scores,
            key=lambda score: (score.valid, score.score if reverse else -score.score),
            reverse=True,
        )


# ---------------------------------------------------------------------------
# Candidate adapters
# ---------------------------------------------------------------------------


def as_action_fn(candidate):
    """
    Normalise a candidate into the `ActionFn` every environment consumes.

    Accepts:
      * an `ActionFn` — `(State, timestep) -> list[ActionOp]`
      * a plan — `list[list[ActionOp]]`, bag `t` applied at timestep `t`, which
        is what the one-shot and evolve methods emit
      * anything exposing `.action_fn` or `.plan`
    """
    if callable(candidate):
        return candidate

    plan = getattr(candidate, "plan", None)

    if plan is None and hasattr(candidate, "action_fn"):
        return candidate.action_fn

    if plan is None:
        plan = candidate

    try:
        bags = list(plan)
    except TypeError as exception:
        raise TypeError(
            f"candidate {candidate!r} is neither callable, a plan (list of action "
            f"bags), nor an object exposing .action_fn / .plan"
        ) from exception

    def action_fn(state, timestep: int):
        # Past the end of a finite plan the strategy simply stops intervening;
        # the cascade keeps running, which is what a t0-only classical algorithm
        # does and how `--baseline` mode is expressed.
        return list(bags[timestep]) if timestep < len(bags) else []

    return action_fn


def candidate_name(candidate, index: int) -> str:
    for attribute in ("name", "algorithm_id", "__name__"):
        value = getattr(candidate, attribute, None)

        if isinstance(value, str) and value:
            return value

    return f"candidate_{index}"


__all__ = [
    "CandidateScore",
    "ScoringContext",
    "WorldModelScorer",
    "as_action_fn",
    "candidate_name",
    "describe",
]
