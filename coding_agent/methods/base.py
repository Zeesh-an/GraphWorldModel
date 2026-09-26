"""OuterLoopMethod contract + shared helpers."""

import time
from dataclasses import replace
from functools import partial
from typing import Protocol

from coding_agent.agent import CodingAgent
from coding_agent.blocking import (
    blocker_set,
    blocking_plan,
    build_negative_cascade,
    edge_weight_caps,
    lever_of,
    proximity_ring,
)
from coding_agent.containment import build_outbreak, removal_plan, removal_set
from coding_agent.credit import planned_action
from coding_agent.epidemic import (
    build_immunization,
    immunization_plan,
)
from coding_agent.executor import (
    StrategyError,
    budget_key,
    call_strategy,
    validate_actions,
)
from coding_agent.localization import (
    LocalizeAnchor,
    bind_predict_marginals,
    evaluate_localizer,
    summarize_localization,
)
from coding_agent.prediction import (
    PredictAnchor,
    bind_forecast_marginals,
    evaluate_predictor,
    expected_popularity,
    summarize_prediction,
)
from coding_agent.reconstruction import (
    ReconstructAnchor,
    bind_step_marginals,
    evaluate_reconstructor,
    summarize_reconstruction,
    transition_logprob,
)
from coding_agent.rounds import adaptive_action_fn, round_batches, round_schedule
from coding_agent.stream import build_stream
from coding_agent.tools import algorithms, primitives
from coding_agent.tools.adaptive_algorithms import adaptive_algorithms
from coding_agent.tools.blocking_algorithms import all_blocking_algorithms
from coding_agent.tools.dismantling_algorithms import dismantling_algorithms
from coding_agent.tools.immunization_algorithms import immunization_algorithms
from coding_agent.tools.localization_algorithms import (
    localization_algorithms,
    localization_scorers,
)
from coding_agent.tools.prediction_algorithms import prediction_algorithms
from coding_agent.tools.reconstruction_algorithms import reconstruction_algorithms
from coding_agent.search_metrics import paired_band
from coding_agent.types import (
    ActionOp,
    GraphInfo,
    Strategy,
    TaskSpec,
    Trajectory,
    improves,
    rank_by,
)

# Ensemble P(infected) below this counts as "unreached" in feedback
unreached_threshold = 0.10
# ...and above this as decisively reached. The gap between the two keeps the
# reference diff from reporting nodes that merely straddle one threshold
reached_threshold = 0.50
max_listed_nodes = 20
max_listed_diff_nodes = 10
max_listed_seeds = 40
max_listed_pairs = 20
max_anchor_reason_chars = 160
max_listed_communities = 8

# One rollout per name at the start of a refinement loop: the score table the
# agent has to top, in the same evaluator it is being judged by. Under an MC
# evaluator these episodes are charged to the arm like any other, so keep the
# list short. RIS members are dropped under LT, where reverse-reachable sets do
# not describe the dynamics.
anchor_algorithms = (
    "high_degree",
    "degree_discount",
    "pagerank_seeds",
    "imm",
    "random_seeds",
)
ic_only_anchors = ("imm",)

# Added to the leaderboard only for an adaptive task. AdaptGreedy is deliberately
# absent: it costs batch x candidates x mc_runs simulations per round, which
# would dominate startup for a table that exists to set a bar, not to be the
# result. Run it as its own --baselines arm when you want its number.
adaptive_anchor_algorithms = ("adapt_epic", "adapt_degree_discount", "static_split")

# The leaderboard for a CONTAINMENT task, which the IM anchors would be nonsense
# for: they return seed sets, and this planner spends its budget on removals.
# `adaptive_degree` heads the list because it is the row that actually has to be
# beaten. `greedy_blocking` is
# deliberately absent for the same reason `adapt_greedy` is: it costs
# k x candidates x mc_runs episodes and would dominate startup for a table that
# exists to set a bar. Run it as its own --baselines arm when you want its number.
dismantling_anchor_algorithms = (
    "adaptive_degree",
    "corehd",
    "collective_influence_removal",
    "netshield",
    "random_removal",
)

# The leaderboard for an EPIDEMIC CONTROL task, per lever. `degree_immunization`
# heads the node list because it is the row that actually has to be beaten
# (RLGN's own Table 2 has Degree tying Eigenvector to within 0.1 on two of five
# graphs). `netshield` and `dava` are the two published methods this task is
# positioned BETWEEN, and running both is the only way their disagreement is
# visible: the spectral method wins the eigendrop and the data-aware one wins the
# attack rate. `acquaintance_immunization` is on the list because the literature
# names it as the row that most embarrasses learned methods, and
# `random_immunization` is not a throwaway floor: the GAP between it and degree is
# Pastor-Satorras & Vespignani's founding result. `mc_greedy_immunization` is
# deliberately absent for the same reason `greedy_blocking` and `adapt_greedy` are.
immunization_anchor_algorithms = {
    "vaccinate": (
        "degree_immunization",
        "netshield",
        "dava",
        "acquaintance_immunization",
        "random_immunization",
    ),
    "quarantine": (
        "degree_immunization",
        "netshield",
        "dava",
        "acquaintance_immunization",
        "random_immunization",
    ),
    "edge_cut": (
        "netmelt",
        "product_degree",
        "frontier_edge_cut",
        "random_edge_cut",
    ),
    "contact_reduce": (
        "netmelt",
        "product_degree",
        "frontier_edge_cut",
        "random_edge_cut",
    ),
}

# The floor every allocation has to beat, and the reference every
# prevented-infections number divides by: the outbreak with nobody dosed. An
# anchor row rather than a private simulation, so it is measured by the same
# evaluator as the arm it calibrates.
no_immunization = "no_immunization"

# The leaderboard for an INVERSE task. `lpsi` heads it because it is the row that
# actually has to be beaten: a 2017
# label-propagation method with no learning beats both SL-VAE and DDMSL on Digg,
# and it sits inside the agent's own expressible space. `resim_greedy` is
# deliberately absent for the same reason `greedy_blocking` and `adapt_greedy`
# are: it re-simulates every candidate on a private simulator and would dominate
# startup for a table that exists to set a bar, not to be the result.
localization_anchor_algorithms = (
    "lpsi",
    "netsleuth",
    "jordan_center",
    "dmp_localize",
    "infected_degree",
    "random_sources",
)

# The leaderboard for a CASCADE RECONSTRUCTION task. `delayed_bfs` heads it
# because it is the row that actually has to be beaten: Xiao SDM'18's methods
# reach node precision > 0.8 from an O(m + k log k) BFS variant, and it sits
# inside the agent's own expressible space. `personalized_pagerank` is second for
# the opposite reason: it is published to BEAT tree sampling on `ca_grqc`
# specifically, so an arm that does not clear it there has demonstrated nothing.
# `observed_only` is Rozenshtein's `Reports` control and the concrete check that
# a trivial decoder scores badly under the chosen reward.
# `mcmc_decode` and `forward_backward` are deliberately absent for the same reason
# `resim_greedy` and `adapt_greedy` are: they call the kernel thousands of times
# per instance and would dominate startup for a table that exists to set a bar.
reconstruction_anchor_algorithms = (
    "delayed_bfs",
    "personalized_pagerank",
    "consistent_tree_wpct",
    "cult",
    "dhrec",
    "observed_only",
    "random_reconstruction",
)

# The leaderboard for a CASCADE PREDICTION task. `szabo_huberman` heads it because
# it is the row that actually has to be beaten: one feature, one line, from 2008,
# and every paper in this literature still prints it as "Feature-S&H". `seismic`
# and `hawkes` are the generative pair, and they are here for their DECLINE
# behaviour as much as their error: published results show Hawkes beating SEISMIC
# on both mean ARE and on how many cascades it can score at all, which is why the
# two should always be reported together. `mean_size` and `persistence` are the
# two floors: under a LOG-space
# error an instance-blind constant is far stronger than intuition suggests, and an
# arm that only ties with them has learned the corpus's size distribution rather
# than anything about the instance. `mc_forward` is deliberately absent for the same
# reason `mcmc_decode` and `resim_greedy` are: it pays kernel calls per instance
# and would dominate startup for a table that exists to set a bar.
prediction_anchor_algorithms = (
    "szabo_huberman",
    "feature_linear",
    "seismic",
    "hawkes",
    "branching_factor",
    "persistence",
    "mean_size",
)

# The leaderboard for an INFLUENCE BLOCKING task, per lever. `proximity` heads the
# counter-seeding list because it is the row that actually has to be beaten, and
# `degree_blocking` is on it for the opposite reason: CLDAG reports the degree
# heuristic fails outright here, so an arm that only beats degree has demonstrated
# nothing. `greedy_prevention` and
# `cmia_o` are deliberately absent: the first re-simulates the whole competitive
# cascade per candidate and would dominate startup, the second runs a Dijkstra per
# candidate per pick. Run either as its own --baselines arm when you want its number.
blocking_anchor_algorithms = {
    "counter_seed": ("proximity", "reverse_blocking", "rps", "degree_blocking", "random_blocking"),
    "node_block": ("imin_lhga", "imin_lsbm", "advanced_greedy", "degree_blocking", "random_blocking"),
    "edge_block": ("kimura_link_blocking", "out_edge_blocking", "random_edge_blocking"),
    "weight_block": ("kimura_link_blocking", "out_edge_blocking", "random_edge_blocking"),
}

# The floor every blocker has to beat, and the reference every prevented-influence
# number divides by: the rumour with nobody stopping it. Reported as an anchor row
# rather than computed on a private simulator, so it is measured by the same
# evaluator as the arm it is calibrating.
no_blocking = "no_blocking"

# Residual-gain feedback is built from reverse-reachable sets, so it is IC-only.
# theta is small relative to what IMM would ask for: this ranks candidates for a
# prompt, it does not select them.
rr_per_node = 10
rr_max_theta = 20_000
rr_max_nodes = 200_000
max_listed_gains = 10


def _communities(graph: GraphInfo) -> dict[int, int]:
    """{node: community_id}, cached on the graph, summarize() runs every turn."""
    if graph._community_labels is None:
        graph._community_labels = primitives.detect_communities(graph)

    return graph._community_labels


def _rr_covers(graph: GraphInfo, diffusion_model: str) -> tuple | None:
    """({node: RR-set indices it reaches}, theta), cached. None when not applicable."""
    if diffusion_model != "IC" or graph.num_nodes > rr_max_nodes:
        return None

    if graph._rr_covers is None:
        theta = min(rr_max_theta, rr_per_node * graph.num_nodes)
        covers = {}

        for index, rr_set in enumerate(
            primitives.batch_reverse_sample(graph, theta=theta, seed=0)
        ):
            for node in rr_set:
                covers.setdefault(int(node), set()).add(index)

        graph._rr_covers = (covers, theta)

    return graph._rr_covers


def _residual_gains(graph: GraphInfo, diffusion_model: str, seeds: list[int]) -> dict:
    """
    Estimated extra spread each node would add on top of `seeds`, in nodes.

    RR-set coverage not already covered by the seed set, scaled by N/theta. This
    is the marginal-gain signal CELF would compute, at RIS cost and without
    touching the metered evaluator.
    """
    covers_and_theta = _rr_covers(graph, diffusion_model)
    if covers_and_theta is None:
        return {}

    covers, theta = covers_and_theta
    covered = set()
    for seed in seeds:
        covered |= covers.get(int(seed), set())

    scale = graph.num_nodes / theta

    return {
        node: len(cover - covered) * scale
        for node, cover in covers.items()
        if node not in seeds
    }


def _seed_lines(seeds: list[int], graph: GraphInfo) -> list[str]:
    labels = _communities(graph)
    listed = ", ".join(
        f"{node}(d={graph.degree(node)},c={labels.get(node, -1)})"
        for node in seeds[:max_listed_seeds]
    )
    overflow = (
        f", … and {len(seeds) - max_listed_seeds} more"
        if len(seeds) > max_listed_seeds
        else ""
    )

    return [
        f"seeds you chose ({len(seeds)}; d=total degree, c=community id): "
        f"{listed}{overflow}"
    ]


def _community_lines(
    seeds: list[int], marginals: list[float] | None, graph: GraphInfo
) -> list[str]:
    labels = _communities(graph)
    members = {}
    for node in range(graph.num_nodes):
        members.setdefault(labels.get(node, -1), []).append(node)

    seeds_per_community = {}
    for node in seeds:
        community_id = labels.get(node, -1)
        seeds_per_community[community_id] = seeds_per_community.get(community_id, 0) + 1

    ranked = sorted(members, key=lambda key: len(members[key]), reverse=True)
    lines = [
        f"community coverage ({len(members)} communities, largest "
        f"{max_listed_communities} shown; reach = mean P(infected) over members):"
    ]

    for community_id in ranked[:max_listed_communities]:
        group = members[community_id]
        reach = (
            f"{sum(marginals[node] for node in group) / len(group):.0%}"
            if marginals is not None
            else "n/a"
        )
        lines.append(
            f"  c{community_id}: {seeds_per_community.get(community_id, 0)} seeds "
            f"/ {len(group)} nodes, reach {reach}"
        )

    unseeded = [
        community_id
        for community_id in ranked
        if community_id not in seeds_per_community and len(members[community_id]) > 1
    ]
    if unseeded:
        lines.append(
            f"  {len(unseeded)} communities got no seed at all "
            f"(largest: {[len(members[key]) for key in unseeded[:5]]} nodes)"
        )

    return lines


def _gain_lines(gains: dict, graph: GraphInfo) -> list[str]:
    if not gains:
        return []

    ranked = sorted(gains, key=lambda node: gains[node], reverse=True)
    listed = ", ".join(
        f"{node}(d={graph.degree(node)}, +{gains[node]:.1f})"
        for node in ranked[:max_listed_gains]
        if gains[node] > 0
    )

    if not listed:
        return ["no unseeded node has meaningful residual gain: the seed set already covers the reachable graph"]

    return [
        f"highest-value nodes you did NOT seed (estimated extra spread in nodes if "
        f"added on top of your seed set, from reverse-reachable sets): {listed}"
    ]


def _containment_lines(
    trajectory: Trajectory, graph: GraphInfo, task: TaskSpec
) -> list[str]:
    """
    What the removal set did, in the terms a blocker actually reasons about.

    `summarize`'s seed-centric diagnostics (residual gain, community coverage,
    unreached nodes) all describe where a cascade SHOULD go next, which is the
    wrong question here: the removals are what to explain, and the outbreak is
    fixed. This replaces them rather than adding to them.
    """
    removed = removal_set(trajectory.actions)
    outbreak = list(task.outbreak)
    lines = []

    if outbreak:
        listed = ", ".join(
            f"{node}(d={graph.degree(node)})" for node in outbreak[:max_listed_nodes]
        )
        lines.append(
            f"OUTBREAK (fixed, not yours to choose; {len(outbreak)} sources): {listed}"
        )

    if removed:
        listed = ", ".join(
            f"{node}(d={graph.degree(node)})" for node in removed[:max_listed_seeds]
        )
        overflow = (
            f", … and {len(removed) - max_listed_seeds} more"
            if len(removed) > max_listed_seeds
            else ""
        )
        lines.append(
            f"nodes you removed ({len(removed)} of budget {task.budget}, in removal "
            f"order): {listed}{overflow}"
        )

        # Distance from the outbreak is the containment-specific diagnostic: a
        # blocker the cascade never reaches did nothing, and there is no way to
        # see that from the spread number alone
        reachable = set(outbreak)
        frontier = set(outbreak)
        rings = {}
        for hop in range(1, 4):
            frontier = {
                neighbour
                for node in frontier
                for neighbour in graph.out_neighbors(node) + graph.in_neighbors(node)
            } - reachable
            reachable |= frontier
            rings[hop] = len({node for node in removed if node in frontier})

        far = len([node for node in removed if node not in reachable])
        lines.append(
            "removal distance from the outbreak: "
            + ", ".join(f"{count} at {hop} hop(s)" for hop, count in rings.items())
            + f", {far} more than 3 hops away (those can only help if the cascade "
            f"reaches that far)"
        )

    if trajectory.final_marginals is not None:
        infected = [
            node
            for node, frequency in enumerate(trajectory.final_marginals)
            if frequency >= reached_threshold
        ]
        infected.sort(key=graph.degree, reverse=True)
        listed = ", ".join(
            f"{node}(d={graph.degree(node)}, P={trajectory.final_marginals[node]:.2f})"
            for node in infected[:max_listed_nodes]
        )
        lines.append(
            f"nodes the cascade still reaches (P(infected)>={reached_threshold:.0%}): "
            f"{len(infected)}/{graph.num_nodes}: top by degree: {listed}"
        )

    return lines


def _blocking_lines(
    trajectory: Trajectory, graph: GraphInfo, task: TaskSpec
) -> list[str]:
    """
    What the blocker set did, in the terms this literature reasons in.

    `summarize`'s seed-centric diagnostics all describe where a cascade SHOULD go
    next, and `_containment_lines`' removal-distance framing assumes the only lever
    is deletion. Neither answers the question a blocker asks, which is whether its
    counter-cascade got anywhere BEFORE the rumour did: the tie-break makes arriving
    second worth exactly nothing.
    """
    lever = lever_of(task)
    spent = blocker_set(trajectory.actions, lever)
    sources = list(task.outbreak)
    lines = []

    if sources:
        listed = ", ".join(
            f"{node}(d={graph.degree(node)})" for node in sources[:max_listed_nodes]
        )
        lines.append(
            f"RUMOUR SEEDS S_N (fixed, not yours to choose; {len(sources)} nodes, "
            f"already committed at t=0): {listed}"
        )

    if spent:
        listed = ", ".join(str(entry) for entry in spent[:max_listed_seeds])
        overflow = (
            f", … and {len(spent) - max_listed_seeds} more"
            if len(spent) > max_listed_seeds
            else ""
        )
        lines.append(
            f"your {lever} picks ({len(spent)} of budget {task.budget}, in order): "
            f"{listed}{overflow}"
        )

    if lever == "counter_seed" and spent and sources:
        # The diagnostic that decides a counter-seeding result: a blocker the rumour
        # reaches first is worth nothing under either tie-break, and hop distance
        # from S_N is the cheapest honest proxy for who arrives first
        ring = proximity_ring(graph, sources, hops=3)
        position = {node: index for index, node in enumerate(ring)}
        hop_one = set(proximity_ring(graph, sources, hops=1))
        inside = [node for node in spent if node in position]
        lines.append(
            f"reachability of your seeds from the rumour: {len(inside)}/{len(spent)} "
            f"sit within 3 hops of S_N ({sum(1 for node in spent if node in hop_one)} "
            f"of them adjacent to a source). A seed the rumour never reaches saves "
            f"nobody, and one it reaches FIRST saves nobody either: prevented "
            f"influence counts only nodes that would otherwise have been infected."
        )

    if trajectory.final_marginals is not None:
        infected = [
            node
            for node, frequency in enumerate(trajectory.final_marginals)
            if frequency >= reached_threshold
        ]
        infected.sort(key=graph.degree, reverse=True)
        listed = ", ".join(
            f"{node}(d={graph.degree(node)}, P={trajectory.final_marginals[node]:.2f})"
            for node in infected[:max_listed_nodes]
        )
        lines.append(
            f"nodes the RUMOUR still reaches (P>={reached_threshold:.0%}): "
            f"{len(infected)}/{graph.num_nodes}: top by degree: {listed}"
        )

    return lines


def summarize(
    trajectory: Trajectory, graph: GraphInfo | None = None, task: TaskSpec | None = None
) -> str:
    # An inverse task rolled out no cascade, so every diagnostic below: frontier
    # counts, residual gain, community reach: describes something that did not
    # happen. Its own summary answers the question that was actually asked: which
    # sources were recovered, which were missed, and what the misses have in common.
    # A DECODER's is different again: the split that matters there is the node half
    # against the tree half, which is the failure the reward is weighted to prevent.
    # A FORECAST task rolled out no cascade either, and its diagnostic is different
    # again: the split that matters is over- against under-prediction, because MSLE
    # is symmetric in log space and a model that misses every viral cascade posts
    # the same number as one that invents virality everywhere.
    if task is not None and task.forecasts and graph is not None:
        return summarize_prediction(trajectory, graph, task)

    if task is not None and task.decodes and graph is not None:
        return summarize_reconstruction(trajectory, graph, task)

    if task is not None and task.recovers and graph is not None:
        return summarize_localization(trajectory, graph, task)

    reward_se = trajectory.cost.get("reward_se", 0.0)
    frontier_counts = [len(state.frontier) for state in trajectory.states[1:]]
    label = (
        "final_rumour_size (LOWER IS BETTER)"
        if task is not None and task.blocks
        else "final_infected (LOWER IS BETTER)"
        if task is not None and task.contains
        else "final_spread"
    )

    lines = [
        f"{label}={trajectory.reward:.2f} (±{reward_se:.2f} SE), "
        f"steps={len(trajectory.infected_counts)}, "
        f"counts={[round(count, 1) for count in trajectory.infected_counts]}"
    ]
    lines.append(f"frontier_counts={frontier_counts}")

    if task is not None and task.blocks:
        # The counter-cascade's own trajectory, which the negative-only counts above
        # cannot show: a blocker whose positive wave never grows spent its budget on
        # nodes with nowhere to spread
        positive = [len(state.pos_infected) for state in trajectory.states[1:]]
        if any(positive):
            lines.append(f"your_counter_cascade_counts={positive}")

    if frontier_counts and frontier_counts[-1] == 0:
        death_step = len(frontier_counts) - 1 - frontier_counts[::-1].index(0)
        lines.append(
            f"cascade dead by t={death_step}: actions scheduled after that did nothing"
        )

    if graph is None:
        return "\n".join(lines)

    if task is not None and task.blocks:
        return "\n".join(lines + _blocking_lines(trajectory, graph, task))

    if task is not None and task.contains:
        return "\n".join(lines + _containment_lines(trajectory, graph, task))

    seeds = sorted(
        {
            action.target
            for bag in trajectory.actions
            for action in bag
            if action.op == "add_node"
        }
    )
    diffusion_model = task.diffusion_model if task is not None else "IC"
    gains = _residual_gains(graph, diffusion_model, seeds)

    if seeds:
        lines += _seed_lines(seeds, graph)

    # One pass over the seeds' out-neighbourhoods rather than a double loop over
    # the seed set: at 20% of a 117K-node graph the plan holds 23K seeds, and the
    # pairwise form did 273M membership tests per summary. The list is capped like
    # every other list here, because printing it whole put one digg prompt past
    # the model's context window
    seed_set = set(seeds)
    adjacent_pairs = sorted(
        {
            (min(first, second), max(first, second))
            for first in seeds
            for second in graph.out_neighbors(first)
            if second != first and second in seed_set
        }
    )
    if adjacent_pairs:
        listed = ", ".join(
            f"({first}, {second})" for first, second in adjacent_pairs[:max_listed_pairs]
        )
        more = (
            f", … and {len(adjacent_pairs) - max_listed_pairs} more"
            if len(adjacent_pairs) > max_listed_pairs
            else ""
        )
        lines.append(
            f"adjacent seed pairs (overlapping neighborhoods, likely redundant "
            f"budget): {len(adjacent_pairs)} pairs: {listed}{more}"
        )

    if trajectory.final_marginals is not None:
        unreached = [
            node
            for node, frequency in enumerate(trajectory.final_marginals)
            if frequency < unreached_threshold
        ]
        # Ranked by residual gain, not degree: a high-degree node the cascade
        # never reaches is usually unreachable, whereas a high-gain one is the
        # node actually worth spending a seed on
        unreached.sort(key=lambda node: gains.get(node, float(graph.degree(node))), reverse=True)
        listed = ", ".join(
            f"{node}(d={graph.degree(node)}"
            + (f", +{gains[node]:.1f}" if node in gains else "")
            + ")"
            for node in unreached[:max_listed_nodes]
        )
        overflow = (
            f", … and {len(unreached) - max_listed_nodes} more"
            if len(unreached) > max_listed_nodes
            else ""
        )
        lines.append(
            f"unreached nodes (P(infected)<{unreached_threshold:.0%} across the "
            f"ensemble): {len(unreached)}/{graph.num_nodes}, top by estimated gain: "
            f"{listed}{overflow}"
        )

    if seeds:
        lines += _community_lines(seeds, trajectory.final_marginals, graph)

    lines += _gain_lines(gains, graph)

    return "\n".join(lines)


def paired_delta(
    trajectory: Trajectory,
    incumbent: Trajectory,
    incumbent_label: str = "your best so far",
    sense: str = "maximize",
    unit: str = "nodes",
) -> str:
    """
    Signed change against the incumbent, with the noise band that decides it.

    Both rollouts run at the same seed, so the realizations are shared and the
    difference is far better resolved than either absolute number, but on a
    hub-dominated graph the whole algorithmic spread can still sit inside this
    band, and the model needs to be told that rather than chase it. The band is
    the paired standard error, the one acceptance reads.

    `sense` decides what a negative delta MEANS. Under containment fewer infected
    nodes is the win, so an unflipped verdict would coach the model to undo every
    improvement it makes.
    """
    delta = trajectory.reward - incumbent.reward
    # The SAME band the acceptance rule reads, so the verdict the model is told
    # and the decision the loop makes can never disagree
    band, _ = paired_band(trajectory, incumbent)

    if abs(delta) <= band:
        verdict = (
            "INSIDE THE NOISE: this change did nothing measurable, so do not "
            "read anything into its sign"
        )
    elif improves(trajectory.reward, incumbent.reward, sense):
        verdict = "a real improvement: keep what caused it"
    else:
        verdict = "a real regression: undo what caused it"

    direction = "fewer is better" if sense == "minimize" else "more is better"
    # F1, the reconstruction score and every prediction error live on scales where a
    # 0.02 move is large, so a fixed 2-decimal format would print every real change
    # as +0.00. Only a NODE COUNT wants two.
    digits = 2 if unit == "nodes" else 4

    return (
        f"CHANGE vs {incumbent_label} ({incumbent.reward:.{digits}f}): "
        f"{delta:+.{digits}f} {unit} ({direction}). The 2-sigma noise band on this "
        f"comparison is ±{band:.{digits}f}, so this is {verdict}."
    )


def validate_plan(plan: list, task: TaskSpec, graph: GraphInfo) -> None:
    """Validate every bag and the whole-plan budgeted total against the task budget."""
    total_units = 0
    targeted = set()
    noun = {
        "add_node": "seeds",
        "remove_node": "removes",
        "remove_edge": "cuts",
        "set_edge_weight": "reweights",
    }.get(task.budget_op, "spends")
    # Only the weight-reduction lever needs the cap, and building it costs one pass
    # over edge_index, so it is computed where it is used rather than carried around
    caps = (
        edge_weight_caps(graph) if task.budget_op == "set_edge_weight" else None
    )

    for timestep, bag in enumerate(plan):
        validate_actions(
            bag,
            graph.num_nodes,
            task.budget,
            task.allowed_ops,
            task.budget_op,
            # A blocking task's `outbreak` is S_N, and its sources are not protected
            # the way a containment task's are: the rumour is ALREADY spreading from
            # them, so cutting one is a legitimate (if usually late) move rather than
            # a way to end the cascade before it starts
            () if task.blocks else task.outbreak,
            edge_weight_caps=caps,
        )

        for action in bag:
            if action.op != task.budget_op:
                continue

            # Re-targeting at a later timestep spends a second unit of budget on
            # something the plan already acted on
            key = budget_key(action)
            if key in targeted:
                raise StrategyError(
                    f"plan {noun} {key} again at t={timestep}; it is already "
                    f"targeted earlier in the plan and the repeat spends budget "
                    f"without changing anything."
                )

            targeted.add(key)
            total_units += 1

    if total_units > task.budget:
        raise StrategyError(
            f"plan {noun} {total_units} nodes in total, exceeds budget {task.budget}"
        )


def attach_context(
    strategy: Strategy, task: TaskSpec, environment: object | None = None
) -> Strategy:
    """
    Hand the strategy what it needs from the task that its signature cannot carry.

    Attributes rather than a signature change because `plan_horizon(graph, budget,
    horizon)`, `act(state, graph, timestep)` and `localize(graph, observation,
    budget)` are the contract every existing method, exemplar and checkpoint is
    written against. `outbreak` is empty for a seeding task, so a generated script
    may read it unconditionally; `budget_op` is what the SCORED harness emits,
    which is `add_node` for seeding and `remove_node` for containment: hardcoding
    it made every scored-mode containment arm fail validation before it was ever
    scored.

    The evaluator bindings (`predict_marginals`, `step_marginals` with
    `transition_logprob`, `forecast_marginals` with `expected_popularity`) are
    attached to CANNED strategies only: the kernel-using members of the library
    pools (`resim_greedy`, `mcmc_decode`, `mc_forward`, ...) read them, and they
    are what makes a condition-1 baseline run on the arm's own evaluator. A
    GENERATED program never gets them: it is offline by construction, and the
    arm's evaluator reaches it only through the reward and the feedback the
    harness computes. This is a deliberate rule (2026-09-04), not a gap.
    """
    strategy.outbreak = tuple(int(node) for node in task.outbreak)
    strategy.budget_op = task.budget_op

    if environment is not None and getattr(strategy, "canned", False):
        strategy.predict_marginals = bind_predict_marginals(environment, task)
        strategy.step_marginals = bind_step_marginals(environment, task)
        strategy.transition_logprob = transition_logprob
        forecast = bind_forecast_marginals(environment, task, seed=task.seed)
        strategy.forecast_marginals = forecast
        strategy.expected_popularity = (
            (lambda adopters, frontier, steps: expected_popularity(
                forecast, adopters, frontier, steps
            ))
            if task.forward_model
            else forecast
        )

    return strategy


def wrap_exogenous(
    action_fn, task: TaskSpec, graph: GraphInfo, stream: object = None
):
    """
    Layer everything the environment must see but the policy is not charged for.

    Every path that builds an ActionFn goes through here: `evaluate_strategy`,
    `per_step`, `windowed`, because a rollout that skips it faces no outbreak and
    no edit stream, and would post a number that looks like a win against arms
    that did face both.

    Applied AFTER validation, deliberately: the outbreak's `add_node` ops would be
    rejected under `--allowed-ops remove_node`, the deletion bag's `remove_edge`
    ops are the mechanics of one removal rather than a second intervention, and
    the stream's edits are exogenous by definition.

    A COMPETITIVE task takes the blocking wrapper instead of the containment one, and
    the difference is not cosmetic: the containment wrapper seeds the outbreak as
    `add_node` ops, and under two cascades `add_node` means the POSITIVE one, so
    reusing it would have every arm start the rumour's own counter-cascade for it.
    The rumour is committed by the simulator's `reset` there, and the wrapper's
    remaining jobs are the detection delay and the removal expansion.

    A COMPARTMENTAL task takes the epidemic wrapper, and that difference is not
    cosmetic either: `vaccinate` expands a bare `remove_node` into the node PLUS its
    incident arcs while `quarantine` expands it into the arcs ALONE, and the
    containment wrapper only knows the first.
    """
    if task.immunizes:
        immunization = build_immunization(graph, task)

        if immunization is not None:
            action_fn = immunization.wrap(action_fn)
    elif task.blocks:
        negative = build_negative_cascade(graph, task)

        if negative is not None:
            action_fn = negative.wrap(action_fn)
    else:
        outbreak = build_outbreak(graph, task)

        if outbreak is not None:
            action_fn = outbreak.wrap(action_fn)

    if stream is not None:
        action_fn = stream.wrap(action_fn)

    return action_fn


def evaluate_strategy(
    strategy: Strategy,
    environment: object,
    task: TaskSpec,
    graph: GraphInfo,
    seed: int | None = None,
) -> tuple[Trajectory, float]:
    """
    Score one generated program, and return how long building its plan took.

    `seed` picks the evaluator's realization for THIS call. `None` is the
    environment's fixed base seed, which is right for a one-off score and wrong
    inside a search: every candidate then faces the identical sampled cascade and
    the loop tunes to its noise. Measured on epidemic_control/primary_school: the
    search reported 40.6 at k=48 on seed 42, the same program re-scored 67-76 on
    seeds 43-45, and the ground-truth referee said 63.7. Search loops pass a
    per-generation seed and `rescore` the incumbent on it, so a comparison is
    paired (common random numbers) within a generation and never across.

    The single point where the three problem families diverge. An INVERSE task
    never rolls out at all: it calls localize() once per labelled episode and
    scores the recovered sets against the truth. A non-adaptive intervention task
    calls plan_horizon() once up front and replays the result; an adaptive one
    calls act() at each round boundary against the state the previous round
    produced. Everything else about the search (the population, the feedback, the
    checkpoints) is identical, which is what makes each contrast an A/B on one
    variable.
    """
    start = time.perf_counter()
    attach_context(strategy, task, environment)

    # A FORECAST task never rolls out either, and gives up one thing the two inverse
    # tasks keep: there is no action to validate, because `a_t` is NULL at every
    # step. It calls predict() once per logged cascade and scores the popularities
    # against what the log says happened.
    if task.forecasts:
        if not task.instances:
            raise StrategyError(
                "no logged cascades were loaded for this forecasting task; the data "
                "stage must have replayed a real corpus for this (dataset, dynamics, "
                "split). Running this on "
                "simulated transitions closes exactly the loop it exists to break."
            )

        # `task.instances` IS the selection pool during search, so a fitted library
        # predictor (S&H's two constants, the ridge, the Hawkes correction) fits on
        # exactly the cascades the search is scored on and never on the held-out
        # ones. `run_experiment` passes the selection pool explicitly when it
        # re-runs the winner on the evaluation split, which is the only path where
        # the two differ.
        return evaluate_predictor(
            strategy, environment, task, graph, list(task.instances),
            fit_examples=list(task.instances),
        )

    if task.decodes:
        if not task.instances:
            raise StrategyError(
                "no labelled episodes were loaded for this decoding task; the data "
                "stage must have run for this (dataset, dynamics, split)"
            )

        return evaluate_reconstructor(
            strategy,
            environment,
            task,
            graph,
            list(task.instances),
            task.tree_weight,
        )

    if task.recovers:
        if not task.instances:
            raise StrategyError(
                "no labelled episodes were loaded for this inverse task; the data "
                "stage must have run for this (dataset, dynamics, split)"
            )

        return evaluate_localizer(
            strategy,
            environment,
            task,
            graph,
            list(task.instances),
            task.source_budget_mode,
        )

    # Built once here and applied to EVERY arm: an arm whose graph moved against
    # one whose graph did not would be measuring the stream, not the method
    stream = build_stream(graph, task.horizon, task.edit_rate, task.seed)

    if task.adaptive:
        batches = round_batches(task.budget, task.rounds, task.per_round_budget)
        # act() runs inside the rollout, so there is no plan to time here; the
        # policy's own compute lands in the environment's rollout_seconds
        action_fn = adaptive_action_fn(strategy, task, graph, batches, stream)
    else:
        # The plan is a function of (graph, budget, horizon) and never of the
        # evaluator's seed, so a re-score of the same program on a fresh
        # realization reuses it: plan_horizon() is the expensive half of an
        # agent iteration (minutes on influence_blocking) and the rollout is
        # seconds
        plan = getattr(strategy, "_plan", None)
        if plan is None:
            plan = call_strategy(strategy.plan_horizon, graph, task.budget, task.horizon)
            validate_plan(plan, task, graph)
            strategy._plan = plan

        action_fn = partial(planned_action, plan)

    action_fn = wrap_exogenous(action_fn, task, graph, stream)

    plan_seconds = time.perf_counter() - start
    trajectory = environment.rollout(action_fn, task.horizon, task.budget, seed=seed)

    return trajectory, plan_seconds


def rescore(
    best: tuple | None,
    environment: object,
    task: TaskSpec,
    graph: GraphInfo,
    seed: int,
) -> tuple | None:
    """
    Re-run the incumbent on the challenger's seed so the two are compared paired.

    A no-op for the three families whose reward is exact (F1, path precision,
    MSLE): there is no realization to overfit, and their evaluation is the
    expensive half of the loop (a localizer pays ~250 s per score on jazz).
    """
    if best is None or task.forecasts or task.decodes or task.recovers:
        return best

    trajectory, _ = evaluate_strategy(best[0], environment, task, graph, seed=seed)

    return (best[0], trajectory)


class _BlockingAnchor:
    """
    Wraps a library blocker as the plan_horizon()-shaped object evaluate_strategy wants.

    Separate from `_PlanAnchor` because the blocking library is the only one whose
    members return two different SHAPES: node ids on three levers and `(u, v)` arcs
    on the fourth, and `blocking_plan` is what reconciles them. `None` as the
    selector is the unopposed reference: an empty plan, so the rumour runs with
    nobody stopping it, which is the number every prevented-influence column divides
    by and the floor every arm has to beat.
    """

    def __init__(self, selector, task: TaskSpec, graph: GraphInfo) -> None:
        self.selector = selector
        self.task = task
        self.graph = graph
        self.source_script = ""

    def plan_horizon(
        self, graph: GraphInfo, budget: int, horizon: int
    ) -> list[list[ActionOp]]:
        if self.selector is None:
            return [[] for _ in range(horizon + 1)]

        picks = self.selector(
            graph,
            budget,
            self.task.diffusion_model,
            negative_seeds=self.task.outbreak,
            horizon=horizon,
        )

        return blocking_plan(
            picks, graph, budget, lever_of(self.task), horizon, self.task.outbreak
        )


class _ImmunizationAnchor:
    """
    Wraps a library immunizer as the plan_horizon()-shaped object evaluate_strategy wants.

    The compartmental twin of `_BlockingAnchor`, and separate from `_PlanAnchor` for
    the same reason: the immunization library's members return two SHAPES, node ids
    on the two node levers and `(u, v)` arcs on the two edge ones, and
    `immunization_plan` is what reconciles them. `None` as the selector is the
    UNPROTECTED reference: an empty plan, so the outbreak runs with nobody dosed,
    which is the number every prevented-infections column divides by.
    """

    def __init__(self, selector, task: TaskSpec, graph: GraphInfo) -> None:
        self.selector = selector
        self.task = task
        self.graph = graph
        self.source_script = ""

    def plan_horizon(
        self, graph: GraphInfo, budget: int, horizon: int
    ) -> list[list[ActionOp]]:
        if self.selector is None:
            return [[] for _ in range(horizon + 1)]

        picks = self.selector(
            graph,
            budget,
            self.task.diffusion_model,
            outbreak=self.task.outbreak,
            horizon=horizon,
        )

        return immunization_plan(
            picks,
            graph,
            budget,
            self.task.epi_lever,
            horizon,
            self.task.outbreak,
            self.task.contact_reduction,
        )


class _PlanAnchor:
    """
    Wraps a static selector as the plan_horizon()-shaped object evaluate_strategy wants.

    Anchors go through `evaluate_strategy` rather than calling `environment.rollout`
    directly so they meet the arm they are setting a bar for under IDENTICAL
    conditions: same outbreak, same edit stream, same removal expansion. An anchor
    rolled out bare would face no outbreak at all on a containment task and post an
    unbeatable zero.
    """

    def __init__(self, selector, task: TaskSpec, graph: GraphInfo) -> None:
        self.selector = selector
        self.task = task
        self.graph = graph
        self.source_script = ""

    def plan_horizon(
        self, graph: GraphInfo, budget: int, horizon: int
    ) -> list[list[ActionOp]]:
        nodes = self.selector(
            graph,
            budget,
            self.task.diffusion_model,
            horizon=horizon,
            outbreak=self.task.outbreak,
        )

        if self.task.contains:
            # Filters the outbreak's sources and tops the set back up, which is
            # what keeps a published dismantler runnable without rewriting it to
            # know an outbreak exists
            return removal_plan(
                nodes, graph, budget, self.task.outbreak, horizon
            )

        return [
            [
                ActionOp(self.task.budget_op, int(node))
                for node in dict.fromkeys(int(node) for node in nodes)
            ]
        ] + [[] for _ in range(horizon)]


class _AdaptiveAnchor:
    """Wraps a per-round policy as the act()-shaped object evaluate_strategy wants."""

    def __init__(self, policy, task: TaskSpec) -> None:
        self.policy = policy
        self.task = task
        self.source_script = ""

    def act(self, state, graph: GraphInfo, timestep: int) -> list[ActionOp]:
        batches = round_batches(
            self.task.budget, self.task.rounds, self.task.per_round_budget
        )
        batch = round_schedule(batches, self.task.round_gap, self.task.horizon).get(
            timestep, 0
        )
        if not batch:
            return []

        return [
            ActionOp("add_node", int(node))
            for node in self.policy(
                state,
                graph,
                batch,
                self.task.diffusion_model,
                total_budget=self.task.budget,
            )
        ]


def accepts(
    candidate: Trajectory,
    incumbent: Trajectory | None,
    sense: str,
    operator: str = "refine",
    candidate_script: str = "",
    incumbent_script: str = "",
) -> bool:
    """
    Whether a candidate replaces the incumbent: its paired delta has to clear the
    standard error of the paired difference (`search_metrics.paired_band`), not
    an epsilon and not the larger marginal standard error.

    Both were scored on the same realization (common random numbers), so the
    band is the noise of that comparison, and a delta inside it is a coin flip
    the search used to take as progress. A `simplify` child is the one exception
    in the other direction: it may replace the incumbent when it is SHORTER and
    not worse beyond the band, which is the parsimony pressure toward algorithms
    a reader can follow.
    """
    if incumbent is None:
        return True

    band, _ = paired_band(candidate, incumbent)
    if improves(candidate.reward, incumbent.reward, sense, band):
        return True

    return (
        operator == "simplify"
        and len(candidate_script) < len(incumbent_script)
        and not improves(incumbent.reward, candidate.reward, sense, band)
    )


def _anchor_rollout(
    name: str,
    strategy: object,
    environment: object,
    task: TaskSpec,
    graph: GraphInfo,
    scored: list,
    failed: list,
) -> None:
    # An anchor is context for the model, not the arm: one library member that
    # times out or fails validation on a large graph (imm at 5% of digg exceeded
    # the 1800 s cap) must cost the leaderboard one row, not the whole arm
    try:
        trajectory, _ = evaluate_strategy(strategy, environment, task, graph)
    except StrategyError as error:
        reason = str(error).splitlines()[0][:max_anchor_reason_chars]
        print(f"[anchor] {name}: not run, {reason}")
        failed.append((name, reason))
        return

    scored.append((name, trajectory))


def _first_anchor(entries: list, failed: list) -> tuple:
    if not entries:
        raise StrategyError(
            "every reference baseline failed, so no leaderboard can be built: "
            + "; ".join(f"{name}: {reason}" for name, reason in failed)
        )

    return entries[0]


def _anchor_text(lines: list[str], failed: list) -> str:
    if failed:
        lines.append(
            "  not run (the bar above is the rows that ran): "
            + "; ".join(f"{name}: {reason}" for name, reason in failed)
        )

    return "\n".join(lines)


def baseline_anchor(
    environment: object, task: TaskSpec, graph: GraphInfo
) -> tuple[str, Trajectory, str]:
    """
    One rollout per classical baseline in the same env: the table to top.

    Returns the leaderboard text, the BEST baseline's trajectory, and its name.
    The trajectory rides along because its per-node marginals are what
    reference_diff() compares against, which costs no further rollouts.
    """
    failed = []

    if task.forecasts:
        # A forecasting task's floor is the classical POPULARITY-PREDICTOR library.
        # These go through `evaluate_strategy` like everything else, which routes
        # them into the prediction path: same cascades, same observation window,
        # same metric as the arm they are setting a bar for.
        scored = []

        for name in prediction_anchor_algorithms:
            predictor = PredictAnchor(name, prediction_algorithms[name], task)
            _anchor_rollout(name, predictor, environment, task, graph, scored, failed)

        scored = rank_by(scored, lambda entry: entry[1].reward, task.sense)
        best_name, best_trajectory = _first_anchor(scored, failed)

        lines = [
            f"REFERENCE SCORES: classical popularity-prediction baselines run on "
            f"THESE cascades, at the same observation window and under the same "
            f"metric. The score is {task.prediction_metric.upper()}, and it is an "
            f"ERROR: **LOWER IS BETTER**, unlike every other task in this repo. "
            f"Beating the top row is the bar. Read the whole table, not just the "
            f"score: `szabo_huberman` is a 2008 one-parameter regression and it is "
            f"the row that matters; `mean_size` ignores the instance entirely, and "
            f"if it is competitive then this corpus's sizes are more predictable "
            f"than its cascades are; `seismic` and `hawkes` DECLINE on supercritical "
            f"cascades, which is why their `declined` column is not zero and why a "
            f"low error there is not automatically a better method:",
        ]
        lines += [
            f"  {name:<20} {task.prediction_metric.upper()} "
            f"{trajectory.reward:8.4f} "
            f"(MALE {trajectory.cost['metrics']['male']:.4f}  "
            f"MAPE {trajectory.cost['metrics']['mape']:.4f}  "
            f"PCC {trajectory.cost['metrics']['pcc']:.4f}  "
            f"declined {trajectory.cost['metrics']['n_failed']:.0f}/"
            f"{trajectory.cost['n_instances']})"
            for name, trajectory in scored
        ]

        return _anchor_text(lines, failed), best_trajectory, best_name

    if task.decodes:
        # A decoding task's floor is the classical TRAJECTORY-DECODER library.
        # These go through `evaluate_strategy` like everything else, which routes
        # them into the reconstruction path: same episodes, same mask, same
        # setting as the arm they are setting a bar for.
        scored = []

        for name in reconstruction_anchor_algorithms:
            decoder = ReconstructAnchor(name, reconstruction_algorithms[name], task)
            _anchor_rollout(name, decoder, environment, task, graph, scored, failed)

        scored = rank_by(scored, lambda entry: entry[1].reward, task.sense)
        best_name, best_trajectory = _first_anchor(scored, failed)

        lines = [
            "REFERENCE SCORES: classical cascade-reconstruction baselines run on "
            "THESE episodes, under the same mask and the same setting. The reward "
            "is the kernel log-likelihood of the decoded history per node minus the "
            "fraction of the observation it contradicts, HIGHER is better, and "
            "beating the top row is the bar. `delayed_bfs` is the row that matters, "
            "and `personalized_pagerank` is on this list because the published "
            "finding is that it BEATS tree sampling on assortative graphs like "
            "ca_grqc. `observed_only` reports exactly what it was shown and infers "
            "nothing: if it is competitive, look at what the kernel expected and it "
            "left out:",
        ]
        lines += [
            f"  {name:<24} reward {trajectory.reward:8.4f} "
            f"(loglik/node {trajectory.cost['metrics']['loglik_per_node']:.4f}  "
            f"consistency {trajectory.cost['metrics']['consistency']:.3f}  "
            f"decoded {trajectory.cost['metrics']['n_predicted']:.1f} nodes)"
            for name, trajectory in scored
        ]

        return _anchor_text(lines, failed), best_trajectory, best_name

    if task.recovers:
        # An inverse task's floor is the classical SOURCE-LOCALIZATION library.
        # These go through `evaluate_strategy` like everything else, which routes
        # them into the localization path: same instances, same k, same
        # observation mode as the arm they are setting a bar for.
        scored = []

        for name in localization_anchor_algorithms:
            localizer = LocalizeAnchor(
                name, localization_algorithms[name], localization_scorers[name], task
            )
            _anchor_rollout(name, localizer, environment, task, graph, scored, failed)

        scored = rank_by(scored, lambda entry: entry[1].reward, task.sense)
        best_name, best_trajectory = _first_anchor(scored, failed)

        lines = [
            "REFERENCE SCORES: classical source-localization baselines run on THESE "
            "episodes, at the same k and the same observation. Consistency = minus "
            "the mean squared error between re-simulating the named sources and the "
            "observation, HIGHER is better (0 is perfect); beating the top row is "
            "the bar. LPSI is the row that matters: it is a 2017 label-propagation "
            "method with no learning at all, and it beats both SL-VAE and DDMSL on "
            "real cascades:"
        ]
        lines += [
            f"  {name:<20} consistency {trajectory.reward:8.5f} "
            f"(over-explained {trajectory.cost['metrics']['n_over']:.1f}  "
            f"under-explained {trajectory.cost['metrics']['n_under']:.1f} nodes "
            f"per episode)"
            for name, trajectory in scored
        ]

        return _anchor_text(lines, failed), best_trajectory, best_name

    if task.blocks:
        # An influence-blocking task's floor is the BLOCKING library, and the row
        # that matters is `no_blocking`: sigma(S_N, empty), the rumour with nobody
        # stopping it. Every other row is only interesting as a difference from it,
        # which is what "prevented influence" means.
        lever = lever_of(task)
        scored = [
            (no_blocking, evaluate_strategy(
                _BlockingAnchor(None, task, graph), environment, task, graph
            )[0])
        ]

        for name in blocking_anchor_algorithms[lever]:
            anchor = _BlockingAnchor(all_blocking_algorithms[name], task, graph)
            _anchor_rollout(name, anchor, environment, task, graph, scored, failed)

        unopposed = scored[0][1].reward
        ranked = rank_by(scored, lambda entry: entry[1].reward, task.sense)
        best_name, best_trajectory = _first_anchor(ranked, failed)

        lines = [
            f"REFERENCE SCORES: classical influence-blocking baselines for the "
            f"{lever} lever, run on THIS graph against THIS rumour, under THIS "
            f"evaluator, at the same budget and horizon. The column is the rumour's "
            f"final size, so LOWER IS BETTER; `prevented` is how much of the "
            f"unopposed cascade each one stopped, which is the quantity every paper "
            f"in this literature reports. `no_blocking` is the rumour with nobody "
            f"stopping it: beating it by a lot is the bar, and the degree heuristic "
            f"is on this list because the published finding is that it FAILS here:",
        ]
        lines += [
            f"  {name:<24} {trajectory.reward:9.2f} "
            f"(±{trajectory.cost.get('reward_se', 0.0):.2f} SE, "
            f"prevented {unopposed - trajectory.reward:+.2f} = "
            f"{100.0 * (unopposed - trajectory.reward) / max(unopposed, 1e-9):.1f}%)"
            for name, trajectory in ranked
        ]

        return _anchor_text(lines, failed), best_trajectory, best_name

    if task.immunizes:
        # An epidemic-control task's floor is the IMMUNIZATION library, and the row
        # that matters is `no_immunization`: the outbreak with nobody dosed. Every
        # other row is only interesting as a difference from it, which is what
        # "prevented infections" means.
        lever = task.epi_lever
        scored = [
            (no_immunization, evaluate_strategy(
                _ImmunizationAnchor(None, task, graph), environment, task, graph
            )[0])
        ]

        for name in immunization_anchor_algorithms[lever]:
            anchor = _ImmunizationAnchor(immunization_algorithms[name], task, graph)
            _anchor_rollout(name, anchor, environment, task, graph, scored, failed)

        unprotected = scored[0][1].reward
        ranked = rank_by(scored, lambda entry: entry[1].reward, task.sense)
        best_name, best_trajectory = _first_anchor(ranked, failed)

        lines = [
            f"REFERENCE SCORES: classical epidemic-control baselines for the "
            f"{lever} lever, run on THIS graph against THIS outbreak, under THIS "
            f"evaluator, at the same budget and horizon. The column is the attack "
            f"rate (how many nodes were EVER infected) so LOWER IS BETTER; "
            f"`prevented` is how many infections each one stopped. "
            f"`no_immunization` is the outbreak with nobody dosed, and beating it "
            f"by a lot is the bar. Read `netshield` against `dava` specifically: "
            f"the first minimizes the graph's leading eigenvalue and does not know "
            f"where the outbreak IS, the second conditions on exactly that, and on "
            f"a small localized outbreak the second usually wins by a wide margin:",
        ]
        lines += [
            f"  {name:<28} {trajectory.reward:9.2f} "
            f"(±{trajectory.cost.get('reward_se', 0.0):.2f} SE, "
            f"prevented {unprotected - trajectory.reward:+.2f} = "
            f"{100.0 * (unprotected - trajectory.reward) / max(unprotected, 1e-9):.1f}%)"
            for name, trajectory in ranked
        ]

        return _anchor_text(lines, failed), best_trajectory, best_name

    if task.contains:
        # A containment task's floor is the DISMANTLING library: the IM anchors
        # return seed sets, and this planner spends its budget on removals
        pool = dismantling_algorithms
        names = list(dismantling_anchor_algorithms)
    else:
        pool = algorithms.algorithms
        names = [
            name
            for name in anchor_algorithms
            if task.diffusion_model == "IC" or name not in ic_only_anchors
        ]

    scored = []

    # A static selector is a static PLAN, so it is scored against a non-adaptive
    # view of the task even when the task has rounds: `evaluate_strategy` would
    # otherwise route it through `adaptive_action_fn` and call an `act()` it does
    # not have. That is also the right reading: a one-shot algorithm dealt at t=0
    # is exactly the control an adaptive arm is measured against.
    static_task = replace(task, rounds=None, per_round_budget=None)

    for name in names:
        selector = _PlanAnchor(pool[name], task, graph)
        _anchor_rollout(name, selector, environment, static_task, graph, scored, failed)

    # Under an adaptive task the static table alone sets the wrong bar: it shows
    # what a one-shot algorithm gets and says nothing about what the published
    # ADAPTIVE algorithms get on the same rounds. AdaptGreedy is the number a
    # generated policy actually has to beat.
    # ...but only on a SEEDING task. The published adaptive policies return seed
    # sets, so on a containment task they emit an op the planner may not use and
    # every anchor rollout would fail validation.
    for name in adaptive_anchor_algorithms if task.adaptive and not task.contains else ():
        policy = _AdaptiveAnchor(adaptive_algorithms[name], task)
        _anchor_rollout(name, policy, environment, task, graph, scored, failed)

    scored = rank_by(scored, lambda entry: entry[1].reward, task.sense)
    best_name, best_trajectory = _first_anchor(scored, failed)

    if task.contains:
        kind = "classical network-dismantling baselines"
        goal = "LOWER is better here; beating the top row is the bar"
    else:
        kind = (
            "classical baselines, static and per-round adaptive,"
            if task.adaptive
            else "classical baselines"
        )
        goal = "beating the top row is the bar"

    lines = [
        f"REFERENCE SCORES: {kind} run on THIS graph, under THIS "
        f"evaluator, at the same budget and horizon. {goal}:"
    ]
    lines += [
        f"  {name:<18} {trajectory.reward:9.2f} "
        f"(±{trajectory.cost.get('reward_se', 0.0):.2f} SE)"
        for name, trajectory in scored
    ]

    return _anchor_text(lines, failed), best_trajectory, best_name


def _diff_node_list(
    nodes: list[int], mine: list[float], theirs: list[float], graph: GraphInfo
) -> str:
    listed = ", ".join(
        f"{node}(d={graph.degree(node)}, ref P={theirs[node]:.2f} vs yours "
        f"{mine[node]:.2f})"
        for node in nodes[:max_listed_diff_nodes]
    )
    overflow = (
        f", … and {len(nodes) - max_listed_diff_nodes} more"
        if len(nodes) > max_listed_diff_nodes
        else ""
    )

    return listed + overflow


def reference_diff(
    trajectory: Trajectory,
    reference: Trajectory | None,
    graph: GraphInfo,
    reference_name: str = "degree_discount",
    sense: str = "maximize",
) -> str | None:
    """
    Where the reference algorithm's cascade went that yours did not, and vice versa.

    Both marginal vectors already exist from rollouts that have been paid for, so
    this is the richest feedback available at zero additional cost.
    """
    # No anchor at all when use_anchor is False, which is every canned arm: the
    # baseline rollouts would be pure cost for a script that never reads a prompt
    if reference is None:
        return None

    mine, theirs = trajectory.final_marginals, reference.final_marginals
    if mine is None or theirs is None:
        return None

    missed = [
        node
        for node in range(graph.num_nodes)
        if theirs[node] >= reached_threshold and mine[node] < unreached_threshold
    ]
    gained = [
        node
        for node in range(graph.num_nodes)
        if mine[node] >= reached_threshold and theirs[node] < unreached_threshold
    ]
    missed.sort(key=graph.degree, reverse=True)
    gained.sort(key=graph.degree, reverse=True)

    lines = [
        f"REFERENCE DIFF (your cascade vs {reference_name}'s: the strongest "
        f"baseline on this graph: same evaluator, per-node P(infected)). Listed "
        f"nodes are DECISIVE flips only (one side >= {reached_threshold:.0%}, the "
        f"other < {unreached_threshold:.0%}); the net line below sums every node, "
        f"so it is larger:"
    ]

    # Same two sets, opposite readings: under containment the nodes the reference
    # reaches and you do not are the ones you successfully PROTECTED
    if sense == "minimize":
        their_reach = f"  it fails to protect {len(missed)} nodes you save"
        your_reach = f"  you lose {len(gained)} nodes it protects"
        net = "  net expected infections vs the reference (negative = you contain more)"
    else:
        their_reach = f"  it reaches {len(missed)} nodes you miss"
        your_reach = f"  you reach {len(gained)} nodes it misses"
        net = "  net expected spread vs the reference"

    if missed:
        lines.append(
            f"{their_reach}: top by degree: "
            f"{_diff_node_list(missed, mine, theirs, graph)}"
        )
    if gained:
        lines.append(
            f"{your_reach}: top by degree: "
            f"{_diff_node_list(gained, mine, theirs, graph)}"
        )
    if not missed and not gained:
        lines.append("  no decisive flips either way")

    lines.append(f"{net}: {sum(mine) - sum(theirs):+.2f} nodes")

    return "\n".join(lines)


class OuterLoopMethod(Protocol):
    def optimize(
        self, agent: CodingAgent, environment: object, task: TaskSpec, graph: GraphInfo
    ) -> tuple[Strategy, Trajectory]: ...
