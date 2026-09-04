"""
Runnable self-check for the epidemic-control contract.

    python -m coding_agent.check_epidemic_control

Everything `research/epidemic_control.md` says must hold, asserted rather than
assumed. The checks are grouped by what would break if one failed, and the two
groups that matter most are the first and the last:

  * **The simulator IS a compartmental process.** SIR at `gamma = 1.0` reproduces
    Independent Cascade to MC precision (§2.1: an infectious node transmits once
    and is then spent, which is IC's rule exactly), the infectious set SHRINKS
    under all three dynamics, and SIS returns nodes to susceptible while `ever`
    stays monotone.
  * **The compartment head is EXACT under its oracle form.** §2.4 is the whole
    modelling contribution of this task: `ICTransmissionHead` composes
    `y_inf = infected + (1 - infected) * p_new`, which is monotone by construction
    and provably cannot represent recovery. The replacement is a per-node
    transition matrix, and `structured_oracle` pins it to the simulator's own
    rates, so its five output columns must match the simulator's one-step
    marginals to sampling error. If that check fails, nothing else in this task
    means anything.

Between them sit the harness rules: the four levers, the dose expansion, the
no-dosing-patient-zero rule, the budget accounting, and the eigendrop-vs-attack-rate
disagreement §8.2 trap 1 predicts.
"""

import numpy as np
import torch

from coding_agent.epidemic import (
    build_immunization,
    contact_reduce,
    edge_cut,
    epidemic_metrics,
    expand_immunization,
    immunization_plan,
    quarantine,
    resolve_lever,
    unprotected_reference,
    vaccinate,
    valid_levers,
)
from coding_agent.envs.monte_carlo_env import MonteCarloEnvironment
from coding_agent.envs.world_model_env import WorldModelEnvironment
from coding_agent.executor import StrategyError, build_strategy, validate_actions
from coding_agent.methods.base import evaluate_strategy, validate_plan
from coding_agent.tools.immunization_algorithms import (
    default_immunization_baselines,
    emittable,
    immunization_algorithms,
    immunization_shape,
)
from coding_agent.tools.primitives import mc_simulate_epidemic
from coding_agent.types import ActionOp, GraphInfo, TaskSpec
from data.wm_epidemic import (
    EpidemicConfig,
    EpidemicSimulator,
    endemic_prevalence,
    model_compartments,
    run_epidemic,
)
from data.wm_graphs import make_synthetic_bundle
from data.wm_simulator import Simulator, epidemic_dynamics
from pipeline.tasks import get_task
from world_model.wm_data import (
    build_epidemic_features,
    build_graph_input,
    channels_for,
    edges_to_arrays,
    epidemic_in_channels,
    epidemic_out_channels,
)
from world_model.wm_metrics import epidemic_curve_metrics, spectral_radius
from world_model.wm_model import WorldModel

passed = []
failed = []

# Draws for the two distributional checks. 4,000 puts the standard error of a
# per-node marginal at ~0.008, so a 0.05 tolerance is ~6 SE and a real bug of the
# size these checks look for (a missing recovery term, a wrong incidence
# definition) moves the max error by 0.3 or more.
mc_draws = 4_000
marginal_tolerance = 0.05


def check(name: str, condition: bool, detail: str = "") -> None:
    (passed if condition else failed).append(name)
    print(f"{'ok ' if condition else 'FAIL'} {name}{': ' + detail if detail else ''}")


def build_graph(nodes: int = 140, seed: int = 3) -> GraphInfo:
    bundle = make_synthetic_bundle("ba", 0, num_nodes=nodes, ba_m=2, seed=seed)

    return GraphInfo(
        num_nodes=nodes,
        edge_index=bundle.edge_index.astype(np.int64),
        ic_probs=bundle.ic_probs,
        directed=False,
    )


def build_task(
    graph: GraphInfo, lever: str = vaccinate, dynamics: str = "SIR", budget: int = 8
) -> TaskSpec:
    budget_op, allowed = resolve_lever(lever)
    rng = np.random.default_rng(11)

    return TaskSpec(
        task="epidemic_control",
        objective="Vaccinate to minimize an outbreak.",
        diffusion_model=dynamics,
        budget=budget,
        horizon=12,
        allowed_ops=allowed,
        sense="minimize",
        budget_op=budget_op,
        outbreak=tuple(sorted(int(n) for n in rng.choice(graph.num_nodes, 2, False))),
        remove_semantics="blocked",
        epidemic=True,
        epi_lever=lever,
        epi_gamma=0.3,
    )


# 1. The registry ---------------------------------------------------------------
def check_registry() -> None:
    task = get_task("epidemic_control")

    check("registry_is_runnable", task.runnable, task.status)
    check("registry_is_epidemic", task.epidemic)
    check("registry_minimizes", task.contains)
    check(
        "registry_uses_blocked_semantics",
        task.remove_semantics == "blocked",
        # `spent` would count every dose as infected and bias the attack rate by +k
        task.remove_semantics,
    )
    check(
        "registry_dynamics_are_compartmental",
        set(task.dynamics) == set(epidemic_dynamics),
        str(task.dynamics),
    )
    check(
        "registry_seeds_an_outbreak",
        task.outbreak_pct > 0,
        f"{task.outbreak_pct}% of N",
    )
    check(
        "registry_generates_every_lever_op",
        set(task.gen_action_ops) == {"remove_node", "remove_edge", "set_edge_weight"},
        # One checkpoint has to serve all four levers, so the head must have seen a
        # dose, a cut AND a reweight
        str(task.gen_action_ops),
    )


# 2. The simulator --------------------------------------------------------------
def check_sir_at_gamma_one_is_ic() -> None:
    """
    §2.1's cheapest correctness check, and the reason it is first.

    A node that enters `I` at step `t` transmits once at `t -> t+1` and, at
    `gamma = 1`, leaves `I` in that same step. That is exactly IC's "a newly
    infected `u` attempts each out-neighbour once, then is spent", so the two
    processes are the same process and their final sizes must agree.
    """
    bundle = make_synthetic_bundle("ba", 0, num_nodes=90, ba_m=2, seed=5)
    sources = [0, 1, 2]
    plan = [[ActionOp("add_node", node) for node in sources]]
    runs = 500

    epidemic_totals = [
        len(
            run_epidemic(
                bundle.nx_graph,
                bundle.ic_prob_map,
                "SIR",
                sources,
                plan,
                15,
                seed=run,
                config=EpidemicConfig(beta_scale=1.0, gamma=1.0),
            )[0].infected
        )
        for run in range(runs)
    ]

    ic_totals = []
    for run in range(runs):
        simulator = Simulator(bundle.nx_graph, ic_prob_map=bundle.ic_prob_map, seed=run)
        simulator.reset("IC")
        state = None
        for timestep in range(16):
            state = simulator.advance(plan[timestep] if timestep < len(plan) else [])
            if timestep > 0 and not state.frontier:
                break

        ic_totals.append(len(state.infected))

    epidemic_mean = float(np.mean(epidemic_totals))
    ic_mean = float(np.mean(ic_totals))
    # Two independent samplers, so the difference of two means has SE ~ sqrt(2)/sqrt(n)
    # times the spread; 1.5 nodes on a 90-node graph is well inside that
    check(
        "sir_at_gamma_one_reproduces_ic",
        abs(epidemic_mean - ic_mean) < 1.5,
        f"SIR {epidemic_mean:.2f} vs IC {ic_mean:.2f} over {runs} runs each",
    )


def check_infectious_set_shrinks() -> None:
    """
    The non-monotone property this whole task exists to test (§2.4).

    `infected` (ever) must never shrink under any dynamics; `frontier` (currently
    infectious) MUST shrink under all three, because that is precisely what
    `ICTransmissionHead` cannot represent.
    """
    graph = build_graph()
    bundle = make_synthetic_bundle("ba", 0, num_nodes=graph.num_nodes, ba_m=2, seed=3)

    for dynamics in epidemic_dynamics:
        simulator = EpidemicSimulator(
            bundle.nx_graph,
            bundle.ic_prob_map,
            seed=1,
            config=EpidemicConfig(gamma=0.4, alpha=0.5),
        )
        simulator.reset(dynamics)
        simulator.advance([ActionOp("add_node", node) for node in (0, 1, 2, 3, 4)])

        ever, prevalence = [], []
        for _ in range(14):
            state = simulator.advance([])
            ever.append(len(state.infected))
            prevalence.append(len(state.frontier))

        check(
            f"{dynamics.lower()}_ever_infected_is_monotone",
            all(later >= earlier for earlier, later in zip(ever, ever[1:])),
        )
        check(
            f"{dynamics.lower()}_infectious_set_shrinks",
            any(later < earlier for earlier, later in zip(prevalence, prevalence[1:])),
            f"|I| path {prevalence[:8]}",
        )

    # SIS specifically: a node leaves I and becomes SUSCEPTIBLE, so it is
    # ever-infected while sitting in none of E / I / R
    simulator = EpidemicSimulator(
        bundle.nx_graph, bundle.ic_prob_map, seed=2, config=EpidemicConfig(gamma=0.5)
    )
    simulator.reset("SIS")
    simulator.advance([ActionOp("add_node", node) for node in (0, 1, 2)])
    for _ in range(6):
        simulator.advance([])

    state = simulator.current_state()
    reverted = set(state.infected) - set(state.frontier) - set(state.exposed)
    check(
        "sis_returns_nodes_to_susceptible",
        bool(reverted) and not state.recovered,
        f"{len(reverted)} ever-infected nodes are susceptible again, R is empty",
    )
    check(
        "sis_has_no_recovered_compartment",
        model_compartments["SIS"] == (0, 2),
        str(model_compartments["SIS"]),
    )


def check_vaccination_is_not_recovery() -> None:
    """
    §8.2 trap 7: a node in `R` is still in the graph and still counted; a VACCINATED
    node is neither. Conflating them is what makes "nodes saved" mean two different
    things across two papers.
    """
    bundle = make_synthetic_bundle("ba", 0, num_nodes=60, ba_m=2, seed=4)
    graph = GraphInfo(60, bundle.edge_index.astype(np.int64), bundle.ic_probs, False)
    simulator = EpidemicSimulator(
        bundle.nx_graph, bundle.ic_prob_map, seed=1, config=EpidemicConfig(gamma=1.0)
    )
    simulator.reset("SIR")

    dose = expand_immunization([ActionOp("remove_node", 5)], graph, vaccinate)
    simulator.advance([ActionOp("add_node", 0)] + dose)
    for _ in range(8):
        simulator.advance([])

    state = simulator.current_state()
    check(
        "a_vaccinated_node_is_never_counted",
        5 not in state.infected and 5 not in state.recovered,
    )
    check(
        "recovered_nodes_stay_counted",
        set(state.recovered) <= set(state.infected) and bool(state.recovered),
        f"|R|={len(state.recovered)} all inside the attack set",
    )


def check_contact_reduction_is_graded() -> None:
    """
    The lever §2.2 says NDlib cannot express, and the reason we wrote our own stepper.

    Scaling an arc's `beta` down must produce an outbreak strictly between the
    unmodified one and the fully-cut one. Under NDlib's SIR this test is not even
    writable: there is no per-arc parameter to scale.
    """
    bundle = make_synthetic_bundle("ba", 0, num_nodes=120, ba_m=2, seed=6)
    graph = GraphInfo(120, bundle.edge_index.astype(np.int64), bundle.ic_probs, False)
    outbreak = [4]
    arcs = sorted(
        {
            (min(int(graph.edge_index[0, e]), int(graph.edge_index[1, e])),
             max(int(graph.edge_index[0, e]), int(graph.edge_index[1, e])))
            for e in range(graph.edge_index.shape[1])
        }
    )[:25]

    attacks = {}
    for factor in (1.0, 0.5, 0.0):
        attacks[factor], _ = mc_simulate_epidemic(
            graph,
            outbreak,
            arcs,
            "SIR",
            lever=contact_reduce,
            mc_runs=200,
            horizon=20,
            seed=1,
            config=EpidemicConfig(gamma=0.25),
            contact_reduction=factor,
        )

    check(
        "contact_reduction_is_monotone_in_the_factor",
        attacks[1.0] >= attacks[0.5] >= attacks[0.0],
        f"r=1.0 -> {attacks[1.0]:.1f}, r=0.5 -> {attacks[0.5]:.1f}, "
        f"r=0.0 -> {attacks[0.0]:.1f}",
    )


# 3. The compartment head -------------------------------------------------------
def check_head_matches_simulator() -> None:
    """
    Warning: THE CHECK THIS TASK STANDS ON (§2.4).

    `structured_oracle` pins the transition matrix to the simulator's own rates:
    `q = beta_scale * w`, `gamma_hat = gamma`, `alpha_hat = alpha`. Every one of its
    five output columns must then equal the simulator's own one-step marginal to
    sampling error, under all three dynamics, which is what makes it a genuine
    ground-truth ceiling rather than a well-shaped approximation of one.

    Two failures this specifically catches, both of which happened while building
    it: a head that composes `ever + newly` rather than `ever + (1 - ever) * newly`
    reads correct under SIR and breaks under SIS; and defining incidence as "newly
    joined `ever`" rather than "left `S`" is off by 0.33 under SIS, because a
    re-infected node leaves `S` again while `ever` does not move.
    """
    for dynamics, gamma, alpha in (("SIR", 0.3, 0.5), ("SIS", 0.4, 0.5), ("SEIR", 0.3, 0.5)):
        bundle = make_synthetic_bundle("er", 0, num_nodes=110, er_p=0.06, seed=9)
        num_nodes = bundle.nx_graph.number_of_nodes()
        config = EpidemicConfig(beta_scale=1.0, gamma=gamma, alpha=alpha)

        simulator = EpidemicSimulator(
            bundle.nx_graph, bundle.ic_prob_map, seed=17, config=config
        )
        simulator.reset(dynamics)
        simulator.advance([ActionOp("add_node", node) for node in (1, 4, 9, 22)])
        for _ in range(4):
            simulator.advance([])

        state = simulator.current_state()
        snapshot = simulator.snapshot()
        _, *marginals = simulator.advance_marginal([], mc_draws)
        simulator.restore(snapshot)

        record = {
            "state": state.to_dict(),
            "action": [],
            "next_marginal_infected": {},
            "next_marginal_incidence": {},
            "next_marginal_exposed": {},
            "next_marginal_infectious": {},
            "next_marginal_recovered": {},
        }
        edge_index, edge_weight = edges_to_arrays(simulator.edges)
        features, _ = build_epidemic_features(record, edge_index, num_nodes)
        graph_input = build_graph_input(
            edge_index, edge_weight, num_nodes, dynamics, torch.device("cpu")
        )

        model = WorldModel(
            "gcn",
            in_channels=epidemic_in_channels,
            hidden_dim=8,
            n_layers=1,
            dropout=0.0,
            head_type="structured_oracle",
            diffusion_model=dynamics,
            remove_semantics="blocked",
            epidemic=True,
            epi_beta=1.0,
            epi_gamma=gamma,
            epi_alpha=alpha,
        )
        predicted = (
            torch.sigmoid(model(torch.from_numpy(features), graph_input))
            .detach()
            .numpy()
        )

        errors = []
        for column in range(epidemic_out_channels):
            truth = np.zeros(num_nodes)
            for node, probability in marginals[column].items():
                truth[int(node)] = probability

            errors.append(float(np.abs(predicted[:, column] - truth).max()))

        check(
            f"{dynamics.lower()}_oracle_head_matches_the_simulator",
            max(errors) < marginal_tolerance,
            "max |pred - true| per column [ever, incidence, E, I, R] = "
            + ", ".join(f"{error:.4f}" for error in errors),
        )


def check_head_can_shrink_the_infectious_set() -> None:
    """
    The property `ICTransmissionHead` provably lacks, stated as a test (§2.4).

    Feed the compartment head a state with an infectious node that has NO
    susceptible neighbours left. A monotone head must predict it still infectious;
    this one must predict `I` dropping by `gamma`.
    """
    bundle = make_synthetic_bundle("ba", 0, num_nodes=40, ba_m=2, seed=8)
    num_nodes = 40
    edge_index, edge_weight = edges_to_arrays(
        {
            (int(bundle.edge_index[0, e]), int(bundle.edge_index[1, e])): float(
                bundle.ic_probs[e]
            )
            for e in range(bundle.edge_index.shape[1])
        }
    )
    # Everyone infectious: nothing left to infect, so the only transition is I -> R
    record = {
        "state": {
            "infected": list(range(num_nodes)),
            "frontier": list(range(num_nodes)),
            "exposed": [],
            "recovered": [],
        },
        "action": [],
        "next_marginal_infected": {},
        "next_marginal_incidence": {},
        "next_marginal_exposed": {},
        "next_marginal_infectious": {},
        "next_marginal_recovered": {},
    }
    features, _ = build_epidemic_features(record, edge_index, num_nodes)
    graph_input = build_graph_input(
        edge_index, edge_weight, num_nodes, "SIR", torch.device("cpu")
    )
    model = WorldModel(
        "gcn",
        in_channels=epidemic_in_channels,
        hidden_dim=8,
        n_layers=1,
        dropout=0.0,
        head_type="structured_oracle",
        diffusion_model="SIR",
        remove_semantics="blocked",
        epidemic=True,
        epi_gamma=0.3,
    )
    predicted = torch.sigmoid(model(torch.from_numpy(features), graph_input)).detach().numpy()

    check(
        "the_head_predicts_recovery",
        abs(predicted[:, 3].mean() - 0.7) < 0.01 and abs(predicted[:, 4].mean() - 0.3) < 0.01,
        f"E[next I] = {predicted[:, 3].mean():.3f} (expected 1 - gamma = 0.70), "
        f"E[next R] = {predicted[:, 4].mean():.3f} (expected gamma = 0.30)",
    )
    check(
        "the_attack_set_still_cannot_shrink",
        float(predicted[:, 0].min()) > 0.99,
        f"min E[next ever] = {predicted[:, 0].min():.4f}",
    )


def check_channel_layout() -> None:
    check(
        "epidemic_channels_are_nine_in_five_out",
        channels_for(epidemic=True) == (9, 5),
        str(channels_for(epidemic=True)),
    )
    try:
        channels_for(competitive=True, epidemic=True)
        check("a_dataset_is_never_both_layouts", False)
    except ValueError:
        check("a_dataset_is_never_both_layouts", True)

    try:
        WorldModel(
            "gcn",
            in_channels=9,
            hidden_dim=8,
            n_layers=1,
            head_type="linear",
            diffusion_model="SIR",
            epidemic=True,
        )
        check("a_compartmental_task_has_no_linear_head", False)
    except ValueError as error:
        check("a_compartmental_task_has_no_linear_head", "simplex" in str(error))


# 4. The levers and the harness -------------------------------------------------
def check_levers() -> None:
    check(
        "four_levers_are_declared",
        set(valid_levers) == {vaccinate, quarantine, edge_cut, contact_reduce},
        str(valid_levers),
    )

    for lever in valid_levers:
        budget_op, allowed = resolve_lever(lever)
        check(
            f"{lever}_budgets_exactly_what_it_permits",
            allowed == (budget_op,),
            f"{budget_op} / {allowed}",
        )

    graph = build_graph()

    # Vaccination deletes the node AND its arcs; quarantine deletes only the arcs
    vaccinated = expand_immunization([ActionOp("remove_node", 3)], graph, vaccinate)
    isolated = expand_immunization([ActionOp("remove_node", 3)], graph, quarantine)

    check(
        "vaccinate_expands_to_the_node_plus_its_arcs",
        sum(1 for op in vaccinated if op.op == "remove_node") == 1
        and any(op.op == "remove_edge" for op in vaccinated),
    )
    check(
        "quarantine_expands_to_the_arcs_alone",
        not any(op.op == "remove_node" for op in isolated)
        and len(isolated) == len(vaccinated) - 1,
        f"{len(isolated)} arc cuts, no node deletion",
    )
    check(
        "expansion_is_idempotent",
        len(expand_immunization(vaccinated, graph, vaccinate)) == len(vaccinated),
    )
    check(
        "the_edge_levers_pass_through_untouched",
        expand_immunization([ActionOp("remove_edge", 1, 2)], graph, edge_cut)
        == [ActionOp("remove_edge", 1, 2)],
    )


def check_budget_and_protection() -> None:
    graph = build_graph()
    task = build_task(graph)
    sources = list(task.outbreak)

    # Dosing an index case ends the outbreak rather than controlling it
    try:
        validate_actions(
            [ActionOp("remove_node", sources[0])],
            graph.num_nodes,
            task.budget,
            task.allowed_ops,
            task.budget_op,
            task.outbreak,
        )
        check("dosing_an_index_case_is_rejected", False)
    except StrategyError as error:
        check("dosing_an_index_case_is_rejected", "OUTBREAK SOURCE" in str(error))

    # The budget counts DOSES, never the incident cuts the expansion adds
    bag = expand_immunization(
        [ActionOp("remove_node", node) for node in (11, 12, 13)], graph, vaccinate
    )
    validate_actions(bag, graph.num_nodes, 3, ("remove_node", "remove_edge"), "remove_node")
    check("the_budget_counts_doses_not_incident_cuts", True, f"{len(bag)} ops, 3 doses")

    # A weight lever may only LOWER an arc
    weight_task = build_task(graph, contact_reduce)
    plan = immunization_plan(
        [(int(graph.edge_index[0, 0]), int(graph.edge_index[1, 0]))],
        graph,
        1,
        contact_reduce,
        horizon=2,
        contact_reduction=0.0,
    )
    validate_plan(plan, weight_task, graph)
    check("a_reduction_below_the_arc_probability_is_accepted", True)

    try:
        validate_plan(
            [[ActionOp("set_edge_weight", int(graph.edge_index[0, 0]),
                       int(graph.edge_index[1, 0]), 1.0)], []],
            weight_task,
            graph,
        )
        check("raising_an_arc_is_rejected", False)
    except StrategyError as error:
        check("raising_an_arc_is_rejected", "only REDUCE" in str(error))


def check_library() -> None:
    graph = build_graph()
    outbreak = (7, 41)
    budget = 10

    for name, function in immunization_algorithms.items():
        if name == "mc_greedy_immunization":
            continue

        picks = function(graph, budget, "SIR", outbreak=outbreak)
        shape = immunization_shape[name]
        ok = len(picks) == budget

        if shape == "node":
            ok = ok and not (set(int(p) for p in picks) & set(outbreak))

        check(f"library_{name}_returns_{budget}_{shape}s", ok, f"{len(picks)} picks")

    for lever, pool in default_immunization_baselines.items():
        check(
            f"default_pool_for_{lever}_is_emittable",
            all(emittable(name, lever) for name in pool),
            f"{len(pool)} members",
        )


def check_spectral_disagreement() -> None:
    """
    §8.2 trap 1, measured rather than asserted.

    A method can post the LARGEST eigendrop and prevent the FEWEST infections,
    because `lambda_1` is a global property that says nothing about where the
    outbreak currently is. That gap is DAVA's entire contribution and the reason
    §9.4 says to position this task against DAVA rather than NetShield, so if the
    two columns ever agree perfectly, the surrogate was sufficient and the task had
    nothing to add.
    """
    graph = build_graph(nodes=250, seed=1)
    rng = np.random.default_rng(5)
    outbreak = sorted(int(node) for node in rng.choice(graph.num_nodes, 3, False))
    config = EpidemicConfig(gamma=0.3)
    intact = spectral_radius(graph.edge_index, graph.num_nodes)

    scored = {}
    for name in ("netshield", "dava", "degree_immunization"):
        picks = immunization_algorithms[name](graph, 18, "SIR", outbreak=outbreak)
        attack, _ = mc_simulate_epidemic(
            graph, outbreak, picks, "SIR", mc_runs=120, horizon=20, seed=1, config=config
        )
        drop = intact - spectral_radius(graph.edge_index, graph.num_nodes, picks)
        scored[name] = (drop, attack)

    best_eigendrop = max(scored, key=lambda name: scored[name][0])
    best_attack = min(scored, key=lambda name: scored[name][1])

    check(
        "the_spectral_surrogate_and_the_attack_rate_disagree",
        best_eigendrop != best_attack,
        f"largest eigendrop: {best_eigendrop} ({scored[best_eigendrop][0]:.2f}); "
        f"lowest attack rate: {best_attack} ({scored[best_attack][1]:.1f})",
    )
    # DAVA optimizes a DIFFERENT objective, and that is the assertable claim. It
    # does NOT always win the attack rate: on a graph where an index case is itself
    # a hub, the dominator tree from the superseed has the whole graph as one
    # child-set and there is nothing to dominate, so degree wins outright. Both
    # outcomes are real and both are worth seeing, so the direction is PRINTED and
    # only the structural difference is asserted.
    check(
        "dava_optimizes_something_other_than_the_eigenvalue",
        scored["dava"][0] < scored["netshield"][0],
        f"eigendrops: dava {scored['dava'][0]:.2f} vs netshield "
        f"{scored['netshield'][0]:.2f}, dava spends its budget on the outbreak's "
        f"dominators, not on the graph's hubs",
    )
    ordered = sorted(scored, key=lambda name: scored[name][1])
    print(
        "     attack rates on this instance: "
        + ", ".join(f"{name} {scored[name][1]:.1f}" for name in ordered)
        + "  (which one wins depends on the graph and on where the outbreak is: "
        "that dependence IS the finding)"
    )


def check_curve_metrics() -> None:
    """§2.6's four shape metrics, and the SIS one that replaces final size."""
    metrics = epidemic_curve_metrics([0, 2, 6, 14, 25, 18, 9, 4, 1, 0], 100)

    check("peak_prevalence_is_the_max", metrics["peak_prevalence"] == 25)
    check("time_to_peak_is_its_argmax", metrics["time_to_peak"] == 4)
    check("auc_is_the_sum", metrics["auc_infectious"] == 79)
    check(
        "endemic_prevalence_averages_after_burn_in",
        abs(endemic_prevalence([10] * 10 + [4] * 10, 0.5) - 4.0) < 1e-9,
    )


def check_prevented_infections_and_the_referee() -> None:
    """
    The reported block, end to end on the Monte Carlo evaluator.

    Warning: THE UNPROTECTED REFERENCE MUST BE WRAPPED. The blocking task's
    counterpart can hand its environment a bare empty plan because `S_N` is
    committed by the simulator's `reset`; here the outbreak is INJECTED by the
    wrapper, so an unwrapped empty plan seeds nothing, the reference comes back 0,
    and every prevented-infections number turns negative. It did, once.
    """
    graph = build_graph(nodes=160, seed=2)
    task = build_task(graph, budget=12)
    environment = MonteCarloEnvironment(
        graph,
        "SIR",
        mc_runs=60,
        base_seed=1,
        remove_semantics="blocked",
        epidemic_config=EpidemicConfig(gamma=0.3),
    )

    unprotected, curve = unprotected_reference(
        environment, task, graph, task.horizon, task.budget
    )
    check(
        "the_unprotected_reference_actually_seeds_the_outbreak",
        unprotected > len(task.outbreak),
        f"{unprotected:.1f} infected with nobody dosed",
    )
    check("the_reference_returns_a_prevalence_curve", bool(curve))

    script = """
class DegreeDoses(Strategy):
    def plan_horizon(self, graph, budget, horizon):
        ranked = sorted(
            (n for n in range(graph.num_nodes) if n not in self.outbreak),
            key=graph.degree,
            reverse=True,
        )
        return [[ActionOp("remove_node", int(n)) for n in ranked[:budget]]] + [
            [] for _ in range(horizon)
        ]
"""
    strategy = build_strategy(script)
    trajectory, _ = evaluate_strategy(strategy, environment, task, graph)
    metrics = epidemic_metrics(
        trajectory.reward,
        unprotected,
        task,
        graph,
        trajectory.actions,
        trajectory.prevalence_curve,
    )

    check(
        "a_generated_dose_allocation_prevents_infections",
        metrics["prevented_infections"] > 0,
        f"{metrics['prevented_infections']:+.2f} of {unprotected:.1f}",
    )
    check(
        "the_metrics_block_carries_the_curve_and_the_spectrum",
        "peak_prevalence" in metrics and "spectral" in metrics,
        f"peak {metrics.get('peak_prevalence', 0):.1f}, "
        f"eigendrop {metrics['spectral']['eigendrop_pct']:.1f}%",
    )
    check(
        "the_budget_was_fully_spent_on_doses",
        metrics["n_spent"] == task.budget,
        f"{metrics['n_spent']} of {task.budget}",
    )


def check_oracle_environment_rolls_out() -> None:
    """The `@oracle` arm: the exact head driven as a free-running simulator."""
    graph = build_graph(nodes=140, seed=7)
    task = build_task(graph, budget=10)
    config = EpidemicConfig(gamma=0.3)

    oracle = WorldModelEnvironment.oracle(
        graph,
        "SIR",
        n_samples=30,
        base_seed=1,
        remove_semantics="blocked",
        epidemic=True,
        epi_beta=1.0,
        epi_gamma=0.3,
    )
    monte_carlo = MonteCarloEnvironment(
        graph, "SIR", mc_runs=200, base_seed=1,
        remove_semantics="blocked", epidemic_config=config,
    )

    immunization = build_immunization(graph, task)
    empty = immunization.wrap(lambda state, timestep: [])

    oracle_reward = oracle.rollout(empty, task.horizon, task.budget).reward
    mc_reward = monte_carlo.rollout(empty, task.horizon, task.budget).reward

    # The oracle head is exact per STEP, so a free-running rollout of it must track
    # the simulator's own ensemble to sampling error rather than saturating
    check(
        "the_oracle_rollout_tracks_the_simulator",
        abs(oracle_reward - mc_reward) < 0.15 * max(mc_reward, 1.0),
        f"oracle {oracle_reward:.1f} vs MC {mc_reward:.1f} "
        f"({100.0 * (oracle_reward - mc_reward) / max(mc_reward, 1.0):+.1f}%)",
    )

if __name__ == "__main__":
    print("=" * 72)
    print("EPIDEMIC CONTROL: contract self-check")
    print("=" * 72)

    for section, runner in (
        ("registry", check_registry),
        ("simulator", check_sir_at_gamma_one_is_ic),
        ("simulator", check_infectious_set_shrinks),
        ("simulator", check_vaccination_is_not_recovery),
        ("simulator", check_contact_reduction_is_graded),
        ("head", check_channel_layout),
        ("head", check_head_matches_simulator),
        ("head", check_head_can_shrink_the_infectious_set),
        ("harness", check_levers),
        ("harness", check_budget_and_protection),
        ("library", check_library),
        ("metrics", check_curve_metrics),
        ("metrics", check_spectral_disagreement),
        ("end-to-end", check_prevented_infections_and_the_referee),
        ("end-to-end", check_oracle_environment_rolls_out),
    ):
        print(f"\n--- {section}: {runner.__name__}")
        runner()

    print("\n" + "=" * 72)
    print(f"{len(passed)} passed, {len(failed)} failed")

    if failed:
        print("failed: " + ", ".join(failed))
        raise SystemExit(1)

    print("epidemic-control contract holds")
