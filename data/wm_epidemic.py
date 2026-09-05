"""
Compartmental SIR / SIS / SEIR simulator: the dynamics `epidemic_control` runs on.

Written rather than borrowed, and `research/epidemic_control.md` §2.2 and §9.2 are
the argument. NDlib DOES ship `SIRModel`, `SISModel` and `SEIRModel`, and reusing
them would have been ~60 lines, but all three declare an EMPTY edge-parameter dict
and compare a uniform draw against the single scalar `params['model']['beta']` for
every susceptible neighbour. There is no per-arc transmission probability at all,
and three things die with it:

  * `set_edge_weight` becomes a no-op, which deletes the graded contact-reduction
    lever: the entire social-distancing branch of §3 and DURLECA's whole action
    space.
  * `GraphInput.edge_weight` degenerates to ones, so the encoder and both anchored
    heads lose the one continuous input they have.
  * **`structured_residual` cannot exist**: it is defined as
    `q = sigmoid(logit(w) + MLP(...))` and there is no `w` to anchor on.

The transition rule is four lines. Writing it here recovers all three, and removes
NDlib's global-`np.random` seeding along the way. NDlib stays where its per-edge
threshold API does work, which is IC and LT.

WHAT IS SIMULATED, and the one convention that decides every number:

    p_inf(v) = 1 - prod_{u in I_t, u -> v} (1 - beta_uv)      S -> E (SEIR) or I
    E -> I with probability alpha                             SEIR only
    I -> R with probability gamma                             SIR / SEIR
    I -> S with probability gamma                             SIS

`gamma` is ONE parameter, the rate of LEAVING `I`; only the destination differs by
model. That is deliberate: the `lambda_1 * beta / delta < 1` threshold of Wang et
al. (SRDS'03) and Prakash et al. (ICDM'11) uses `delta` for the leaving-`I` rate
under both SIS and SIR, so splitting it into two flags would let a run report a
virus strength its own simulator did not use.

**Synchronous, from a pre-step snapshot.** Every node's arrivals and transitions
for one step are computed against `I_t` and only then committed, exactly as
`CompetitiveSimulator` does and for the same reason: NDlib's own models iterate
nodes in dict order, which is harmless for one compartment and would leak an
implicit, unrecorded ordering bias across four.

**SIR at gamma = 1.0 IS Independent Cascade.** A node that enters `I` at step `t`
transmits once at the step `t -> t + 1` and leaves `I` in that same step, which is
exactly IC's "newly infected `u` attempts each out-neighbour once, then is spent".
`coding_agent/check_epidemic_control.py` measures that against NDlib rather than
asserting it, and it is the cheapest correctness check this file has.

`beta_uv = clip(beta_scale * p(u -> v), 0, 1)`, where `p` is the graph's own
weighted-cascade probability `1 / in_deg(v)`. So `--epi-beta 1.0` is the IC
probability unchanged, `--prob-model uniform` recovers the literature's scalar-beta
regime, and `set_edge_weight` writes `beta_uv` directly, which is what makes
graded contact reduction expressible at all.
"""

from dataclasses import dataclass
import networkx as nx
import numpy as np

from data.wm_simulator import (
    ActionOp,
    State,
    blocked,
    epidemic_dynamics,
    valid_remove_semantics,
)

# Compartment codes. Ordered so the model's target columns and the simulator agree
# on one layout, and so `S` is 0 the way every NDlib status vector has it.
susceptible = 0
exposed = 1
infectious = 2
recovered = 3
compartment_names = ("S", "E", "I", "R")

# Which compartments each dynamics can occupy. SIS has no `R` and SIR/SIS no `E`,
# and the head reads this back so its transition matrix has no unreachable rows.
model_compartments = {
    "SIR": (susceptible, infectious, recovered),
    "SIS": (susceptible, infectious),
    "SEIR": (susceptible, exposed, infectious, recovered),
}

# Steps of burn-in discarded before the endemic prevalence is time-averaged.
# SIS has no absorbing state, so "final size" is undefined and the reported
# quantity is `lim |I(t)| / N` (research/epidemic_control.md §8.2 trap 6); a
# fraction rather than a count so it scales with the horizon.
default_burn_in = 0.5


@dataclass()
class EpidemicConfig:
    """Everything about the compartmental dynamics that a run has to record."""

    # Multiplier on the graph's own per-arc probability. 1.0 leaves it at the
    # weighted-cascade value 1 / in_deg(v); the literature's scalar-beta regime is
    # `--prob-model uniform --uniform-p <beta>` with this left at 1.0.
    beta_scale: float = 1.0
    # Rate of LEAVING I: recovery under SIR/SEIR, return-to-susceptible under SIS.
    # 1.0 under SIR reproduces IC exactly.
    gamma: float = 0.3
    # E -> I, SEIR only. Ignored by the other two.
    alpha: float = 0.5
    remove_semantics: str = blocked
    burn_in: float = default_burn_in

    def resolved(self, diffusion_model: str) -> dict:
        if diffusion_model not in model_compartments:
            raise ValueError(
                f"unknown compartmental model {diffusion_model!r}; choose one of "
                f"{epidemic_dynamics}"
            )

        return {
            "compartments": [
                compartment_names[code] for code in model_compartments[diffusion_model]
            ],
            "beta_scale": float(self.beta_scale),
            "gamma": float(self.gamma),
            # Inert outside SEIR, and recorded as None there rather than as a
            # number nothing used
            "alpha": float(self.alpha) if diffusion_model == "SEIR" else None,
            "remove_semantics": self.remove_semantics,
            "burn_in": float(self.burn_in),
        }


class EpidemicSimulator:
    """
    Synchronous discrete-time SIR / SIS / SEIR, with the same `advance(bag) -> State`
    contract the single-cascade `Simulator` has.

    `add_node` seeds an INDEX CASE: the node enters `I` directly, not `E`. Under
    SEIR that is the standard convention (a seeded outbreak is already infectious)
    and it is also what keeps the t=0 action bag readable as the source set by
    `wm_data.load_episode_endpoints`, exactly as it is for every other task.

    `remove_node` is VACCINATION and is `blocked` throughout, as every containment
    task needs: the node leaves the graph, is not counted in the attack set, and
    can neither transmit nor be infected. §8.2 trap 7 is why that matters here more
    than elsewhere: a node in `R` is still in the graph and still occupied a dose
    it did not need, so "recovered" and "removed" are different objects and the
    simulator keeps them apart.
    """

    def __init__(
        self,
        graph: nx.Graph | nx.DiGraph,
        ic_prob_map: dict,
        seed: int = 0,
        config: EpidemicConfig | None = None,
    ) -> None:
        config = config or EpidemicConfig()

        if config.remove_semantics not in valid_remove_semantics:
            raise ValueError(
                f"unknown remove_semantics {config.remove_semantics!r}; "
                f"choose one of {valid_remove_semantics}"
            )

        self.graph = graph
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.num_nodes = graph.number_of_nodes()
        self.model_name = None

        # {(u, v): beta}, the per-arc transmission probability the edge ops
        # mutate. Copied per reset so an episode's edits never reach the bundle.
        self.base_edges = {
            (int(source), int(target)): self._beta(probability)
            for (source, target), probability in (ic_prob_map or {}).items()
        }
        self.edges = {}
        self.out_edges = {}

        # Compartment membership. `ever` is the attack set and is monotone; the
        # other three are the current compartments and are not.
        self.ever = set()
        self.exposed = set()
        self.infectious = set()
        self.recovered = set()
        self.blocked = set()
        # |I| after each committed step, so the endemic prevalence and the peak are
        # readable off one episode without a second pass
        self.prevalence = []

    def _beta(self, probability: float) -> float:
        return float(min(1.0, max(0.0, self.config.beta_scale * float(probability))))

    # Setup
    def reset(self, model_name: str, sources: list[int] | tuple = ()) -> None:
        if model_name not in model_compartments:
            raise ValueError(
                f"unknown compartmental model {model_name!r}; choose one of "
                f"{epidemic_dynamics}"
            )

        self.model_name = model_name
        self.edges = dict(self.base_edges)
        self._rebuild_adjacency()

        self.ever = {int(node) for node in sources}
        self.infectious = set(self.ever)
        self.exposed = set()
        self.recovered = set()
        self.blocked = set()
        self.prevalence = [float(len(self.infectious))]

    def _rebuild_adjacency(self) -> None:
        self.out_edges = {node: [] for node in range(self.num_nodes)}

        for (source, target), probability in self.edges.items():
            self.out_edges[source].append((target, probability))

    # State
    def current_state(self) -> State:
        return State(
            infected=sorted(self.ever),
            frontier=sorted(self.infectious),
            exposed=sorted(self.exposed),
            recovered=sorted(self.recovered),
        )

    def susceptible_nodes(self) -> set[int]:
        """`S` = everything the four current compartments and the vaccine leave over."""
        return (
            set(range(self.num_nodes))
            - self.exposed
            - self.infectious
            - self.recovered
            - self.blocked
        )

    def apply_actions(self, bag: list[ActionOp]) -> None:
        """
        The five ops on a compartmental state.

        `add_node` is an index case (straight into `I`); `remove_node` is a dose,
        and under `blocked` it deletes the node from every compartment so it can
        neither be counted nor transmit. `set_edge_weight` writes `beta_uv`
        directly, which is the graded contact-reduction lever NDlib cannot express.
        """
        rebuild = False

        for action in bag:
            target = int(action.target)

            if action.op == "add_node":
                if target in self.blocked:
                    continue

                self.ever.add(target)
                self.infectious.add(target)
                self.exposed.discard(target)
                self.recovered.discard(target)
            elif action.op == "remove_node":
                if self.config.remove_semantics == blocked:
                    self.blocked.add(target)
                    self.ever.discard(target)

                self.exposed.discard(target)
                self.infectious.discard(target)
                self.recovered.discard(target)
            elif action.op == "add_edge":
                self.edges[(target, int(action.destination))] = self._beta(
                    action.weight if action.weight is not None else 1.0
                )
                rebuild = True
            elif action.op == "remove_edge":
                self.edges.pop((target, int(action.destination)), None)
                rebuild = True
            elif action.op == "set_edge_weight":
                edge = (target, int(action.destination))
                if edge in self.edges:
                    # NOT scaled by beta_scale: a reweight names the transmission
                    # probability it wants, and re-applying the scale would make
                    # the same op mean different things at different --epi-beta
                    self.edges[edge] = float(min(1.0, max(0.0, float(action.weight))))
                    rebuild = True

        if rebuild:
            self._rebuild_adjacency()

    # Dynamics
    def _arrivals(self) -> set[int]:
        """Which susceptible nodes `I_t` reached this step, from the pre-step snapshot."""
        reached = set()
        susceptibles = self.susceptible_nodes()

        # SORTED, not set order: a set of ints iterates in a hash order that
        # depends on its insertion history, so two runs that reach the same
        # compartment by different paths would consume the RNG stream differently
        # and a replayed seed would not reproduce its own number
        for source in sorted(self.infectious):
            if source in self.blocked:
                continue

            for target, probability in self.out_edges[source]:
                if (
                    target in susceptibles
                    and target not in reached
                    and self.rng.random() < probability
                ):
                    reached.add(target)

        return reached

    def _step(self) -> None:
        """One synchronous compartment update, committed from a pre-step snapshot."""
        arrivals = self._arrivals()

        # I -> R (SIR/SEIR) or I -> S (SIS). Drawn against the SAME I_t that
        # transmitted, so a node can transmit and leave in one step: the standard
        # discrete-time convention and the one NDlib's own SIR uses.
        leaving = {
            node
            for node in sorted(self.infectious)
            if self.rng.random() < self.config.gamma
        }

        if self.model_name == "SEIR":
            promoted = {
                node
                for node in sorted(self.exposed)
                if self.rng.random() < self.config.alpha
            }
            self.exposed = (self.exposed - promoted) | arrivals
            self.infectious = (self.infectious - leaving) | promoted
        else:
            self.exposed = set()
            self.infectious = (self.infectious - leaving) | arrivals

        if self.model_name == "SIS":
            # No absorbing compartment: a node that leaves I is susceptible again
            # and may be re-infected later. `ever` still grows monotonically.
            self.recovered = set()
        else:
            self.recovered |= leaving

        self.ever |= arrivals
        self.ever -= self.blocked
        self.prevalence.append(float(len(self.infectious)))

    def advance(self, bag: list[ActionOp]) -> State:
        """s_{t+1} = T_endo(T_exo(s_t, a_t))."""
        self.apply_actions(bag)
        self._step()

        return self.current_state()

    def advance_marginal(
        self, bag: list[ActionOp], num_mc: int
    ) -> tuple[State, dict, dict, dict, dict, dict]:
        """
        One-step marginals for the five soft targets the compartment head is fit on.

        Same shape as `Simulator.advance_marginal` with five count dicts instead of
        two: `ever`, `incidence` (who left `S` this step), and the three current
        compartments. The action is applied ONCE (it is deterministic) and only
        the compartment update is redrawn.

        All three dynamics are stochastic (they draw against `beta`, `gamma` and
        `alpha` per step exactly as IC draws against `p`), so all three take the
        full `num_mc` draws. §2.1 item 5 records that reusing IC's
        `draws = num_mc if model == "IC" else 1` line here would collapse every
        target onto a single realization.
        """
        if num_mc < 1:
            raise ValueError(f"num_mc must be >= 1, got {num_mc}")

        self.apply_actions(bag)
        post_action = self.snapshot()
        # INCIDENCE is "who left S this step", which under SIS is NOT the same as
        # "who newly joined `ever`": a node that recovered back to susceptible and
        # is re-infected leaves S again while `ever` does not move. The
        # epidemiological definition is the one the head composes
        # (`susceptible * p_inf`), so it is the one recorded here: measured at 0.33
        # max error against the head before the distinction was made.
        susceptible_before = self.susceptible_nodes()

        counts = [{}, {}, {}, {}, {}]
        last_state = None

        for _ in range(num_mc):
            self.restore(post_action)
            self._step()

            groups = (
                set(self.ever),
                susceptible_before & (self.exposed | self.infectious),
                set(self.exposed),
                set(self.infectious),
                set(self.recovered),
            )
            for table, group in zip(counts, groups, strict=True):
                for node in group:
                    table[node] = table.get(node, 0) + 1

            last_state = self.current_state()

        marginals = [
            {int(node): count / num_mc for node, count in table.items()}
            for table in counts
        ]

        # The episode continues down the LAST realized path while the targets
        # describe the distribution over all of them, matching both other simulators
        self.ever = set(last_state.infected)
        self.infectious = set(last_state.frontier)
        self.exposed = set(last_state.exposed)
        self.recovered = set(last_state.recovered)

        return (last_state, *marginals)

    def set_state(
        self,
        infected,
        frontier,
        exposed_nodes=(),
        recovered_nodes=(),
    ) -> None:
        """
        Force the model into an ARBITRARY mid-epidemic state.

        The compartmental twin of `Simulator.set_state`, and it exists for the same
        caller: a kernel evaluation at a proposed state rather than a replay of one
        the simulator visited. The four sets are written verbatim and `ever` is
        widened to cover them, because a node currently in `E`, `I` or `R` is
        ever-infected by definition and a caller that forgot to say so would
        otherwise produce a state this simulator can never reach.
        """
        self.infectious = {int(node) for node in frontier} - self.blocked
        self.exposed = {int(node) for node in exposed_nodes} - self.blocked
        self.recovered = {int(node) for node in recovered_nodes} - self.blocked
        self.ever = (
            {int(node) for node in infected}
            | self.infectious
            | self.exposed
            | self.recovered
        ) - self.blocked

    # Fork support
    def snapshot(self) -> tuple:
        return (
            set(self.ever),
            set(self.exposed),
            set(self.infectious),
            set(self.recovered),
            set(self.blocked),
            dict(self.edges),
            list(self.prevalence),
        )

    def restore(self, snapshot: tuple) -> None:
        (
            self.ever,
            self.exposed,
            self.infectious,
            self.recovered,
            self.blocked,
        ) = (set(snapshot[index]) for index in range(5))
        edges, self.prevalence = dict(snapshot[5]), list(snapshot[6])

        # The graph lives HERE rather than inside NDlib, so a restore puts the edge
        # table back too and no `revert_edges` companion is needed
        if edges != self.edges:
            self.edges = edges
            self._rebuild_adjacency()


def endemic_prevalence(curve: list[float], burn_in: float = default_burn_in) -> float:
    """
    Time-averaged `|I(t)|` after burn-in: the metric SIS needs and final size is not.

    §8.2 trap 6: SIS has no terminal state, so "final epidemic size" is undefined
    and the quantity the literature reports is `lim_t |I(t)| / N`, estimated by
    averaging after the transient. Reported for every dynamics rather than only SIS,
    because under SIR it is a legitimate (and near-zero) description of the tail and
    a column that changes meaning per row is worse than one that does not.
    """
    if not curve:
        return 0.0

    start = min(int(len(curve) * max(0.0, min(1.0, burn_in))), len(curve) - 1)

    return float(np.mean(curve[start:]))


def run_epidemic(
    graph: nx.Graph | nx.DiGraph,
    ic_prob_map: dict,
    diffusion_model: str,
    sources: list[int],
    plan: list[list[ActionOp]],
    horizon: int,
    seed: int = 0,
    config: EpidemicConfig | None = None,
) -> tuple[State, list[float]]:
    """
    One epidemic episode start to finish: seed the outbreak, apply `plan[t]`, run `horizon` steps.

    The shared driver behind the MC environment, the immunization library's own
    spread estimator and the self-check, so all three agree on what "run this
    vaccination set" means down to the recovery draw. Returns the terminal state and
    the prevalence curve, because the shape of the outbreak is half of what this
    task is graded on (§2.6) and recomputing it would need a second run.

    The early break needs BOTH an empty infectious set and an empty exposed one:
    under SEIR a latent node with nobody left infectious is still going to become
    infectious, so breaking on `I` alone would truncate the epidemic.
    """
    simulator = EpidemicSimulator(graph, ic_prob_map, seed=seed, config=config)
    simulator.reset(diffusion_model, sources)
    state = simulator.current_state()

    for timestep in range(horizon + 1):
        bag = plan[timestep] if timestep < len(plan) else []
        state = simulator.advance(bag)

        if timestep > 0 and not state.frontier and not state.exposed and not bag:
            break

    return state, list(simulator.prevalence)
