"""
Runnable self-check for the cascade-reconstruction contract.

    python -m coding_agent.check_cascade_reconstruction

Nine checks, each one a claim `research/cascade_reconstruction.md` makes that would
be expensive to discover was false halfway through a sweep. They run on a tiny
generated dataset in a temp directory and take a few seconds; nothing here needs a
trained world model or an LLM.

  1. **The transmission edge exists and is real.** NDlib does not emit one (§9 item
     9), so the traced models are ours, and a traced model that silently changed
     the dynamics would invalidate every episode. Checked against the untraced
     model's own distribution, and every recorded `(u, v)` checked to be an arc.
  2. **A source has no parent and activates at t = 0.** The whole `parent = None`
     convention rests on it.
  3. **Every masking setting produces the observation it claims.** Four settings,
     four protocols (§8.3), and a mislabelled one would run a different experiment
     under the reported name.
  4. **The four kernel bindings differ, and @native raises.** Conditions 3-6 are an
     ablation on one variable (§2.5.2); if the bindings were the same object the
     ladder would measure nothing.
  5. **The reward is not gameable by the easy half.** §2.6, and §2.11 risk 1's
     explicit instruction: a trivial decoder must score badly.
  6. **...nor by naming almost nothing.** The second gaming corner, which the
     literature does not name because no published method has a search.
  7. **The contract rejects an incoherent trajectory.** A parent that is not an
     arc, a source at t > 0, a node at t > 0 with no parent.
  8. **`transition_logprob` agrees with the kernel it derives from.** One kernel
     call behind both (§2.5.2), so a disagreement means one of them is wrong.
  9. **Every library decoder returns a valid trajectory on every setting.** The
     condition-1 pool is the bar; a member that raises is a missing row, not a
     weak one.
"""

import json
import shutil
import tempfile
from pathlib import Path
import numpy as np

from coding_agent.envs.monte_carlo_env import MonteCarloEnvironment
from coding_agent.executor import StrategyError, build_strategy
from coding_agent.reconstruction import (
    CascadeInstance,
    Observation,
    evaluate_reconstructor,
    final_snapshot,
    hidden_nodes,
    load_cascades,
    partial_nodes,
    partial_times,
    reconstruction_label_metrics,
    transition_logprob,
    trivial_decoder_reward,
    unavailable_step_marginals,
    valid_settings,
    validate_reconstruction,
)
from coding_agent.tools.reconstruction_algorithms import reconstruction_algorithms
from coding_agent.types import GraphInfo, State, TaskSpec
from data.generate_wm_data import GenConfig, run_generation
from data.wm_simulator import ActionOp, Simulator
from world_model.wm_data import load_graph_store
from world_model.wm_metrics import reconstruction_metrics, reconstruction_reward

# Small enough to finish in seconds, big enough for a cascade with a real tree
check_nodes = 80
check_edge_p = 0.06
check_rollouts = 12
check_horizon = 6
check_instances = 6

# Draws for the traced-vs-untraced distribution check. A traced model that changed
# the dynamics would show up as a mean-spread gap; 200 draws resolves one of ~2
# nodes on this graph, which is far tighter than any plausible bug.
distribution_draws = 200
distribution_tolerance = 0.15

passed = []
failed = []


def check(name: str, condition: bool, detail: str = "") -> None:
    (passed if condition else failed).append(f"{name}: {detail}" if detail else name)
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def generate(out_dir: Path) -> dict:
    return run_generation(
        GenConfig(
            dataset="er",
            num_graphs=1,
            # One tiny in-graph dataset: a disjoint-by-graph split is impossible
            # and irrelevant here, so opt into the per-episode draw explicitly
            split_mode="episode_random",
            syn_nodes=check_nodes,
            er_p=check_edge_p,
            models=["IC"],
            prob_model="weighted",
            uniform_p=0.1,
            budget=5,
            budget_pct=None,
            budget_pct_range=(5.0, 15.0),
            algorithms=["random", "degree"],
            rollouts=check_rollouts,
            horizon=check_horizon,
            inject_p=0.0,
            action_ops=[],
            weight_lo=0.0,
            weight_hi=1.0,
            cf_prob=0.0,
            cf_branches=0,
            split=(0.7, 0.15, 0.15),
            seed=7,
            mc_marginals=4,
            out_dir=str(out_dir),
            trace_parents=True,
        )
    )


def check_traced_dynamics(graph: GraphInfo) -> None:
    """The traced model must record real arcs WITHOUT changing what it simulates."""
    import networkx as nx

    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(graph.num_nodes))
    probabilities = {}
    for edge in range(graph.edge_index.shape[1]):
        source = int(graph.edge_index[0, edge])
        target = int(graph.edge_index[1, edge])
        nx_graph.add_edge(source, target)
        probabilities[(source, target)] = float(graph.ic_probs[edge])

    seeds = [ActionOp("add_node", node) for node in range(5)]
    spreads = {}

    for traced in (False, True):
        totals = []
        for draw in range(distribution_draws):
            simulator = Simulator(
                nx_graph, probabilities, seed=draw, trace_parents=traced
            )
            simulator.reset("IC")
            state = simulator.advance(seeds)
            for _ in range(check_horizon):
                state = simulator.advance([])
            totals.append(len(state.infected))

        spreads[traced] = float(np.mean(totals))

    gap = abs(spreads[True] - spreads[False]) / max(spreads[False], 1.0)
    check(
        "traced IC matches untraced IC in distribution",
        gap <= distribution_tolerance,
        f"traced {spreads[True]:.2f} vs untraced {spreads[False]:.2f} over "
        f"{distribution_draws} draws ({100 * gap:.1f}% apart)",
    )


def check_parents_are_arcs(data_dir: Path, graph: GraphInfo) -> None:
    """Every recorded (u, v) has to be an arc, and every source has to be root."""
    records = [
        json.loads(line)
        for line in (data_dir / "transitions_IC_train.jsonl").read_text().splitlines()
        if line.strip()
    ]
    bad_arcs = []
    total = 0

    for record in records:
        for node, causes in (record.get("parents") or {}).items():
            for cause in causes:
                total += 1
                if int(cause) not in graph.in_neighbors(int(node)):
                    bad_arcs.append((cause, node))

    check(
        "every recorded transmission travels along an arc that exists",
        not bad_arcs and total > 0,
        f"{total} transmissions recorded, {len(bad_arcs)} off-graph",
    )

    roots = [
        (record["episode_id"], node)
        for record in records
        if record["t"] == 0
        for action in record["action"]
        if action["op"] == "add_node"
        for node in [str(action["target"])]
        if (record.get("parents") or {}).get(node) != []
    ]
    check(
        "a source is recorded with an EMPTY parent list",
        not roots,
        f"{len(roots)} seeded nodes carry a parent they should not",
    )


def check_settings(data_dir: Path) -> None:
    """Each of the four masks has to produce the observation it advertises."""
    common = dict(limit=check_instances, seed=3, observation_rate=0.4, hidden_rate=0.2)

    times = load_cascades(str(data_dir), "IC", "train", setting=partial_times, **common)
    check(
        "partial_times reports activation times",
        all(instance.observation.times for instance in times),
        f"{sum(len(i.observation.times) for i in times)} timed reports",
    )

    nodes = load_cascades(str(data_dir), "IC", "train", setting=partial_nodes, **common)
    check(
        "partial_nodes withholds every activation time",
        all(not instance.observation.times for instance in nodes)
        and all(instance.observation.reported for instance in nodes),
        "reports present, times empty",
    )

    snapshot = load_cascades(
        str(data_dir), "IC", "train", setting=final_snapshot, **common
    )
    check(
        "final_snapshot gives the terminal state and NO reports",
        all(not instance.observation.reported for instance in snapshot)
        and all(instance.observation.final_state is not None for instance in snapshot),
        "reports empty, final_state present",
    )

    hidden = load_cascades(str(data_dir), "IC", "train", setting=hidden_nodes, **common)
    invisible = [
        int((~instance.observation.visible).sum())
        for instance in hidden
        if instance.observation.visible is not None
    ]
    sources_hidden = [
        node
        for instance in hidden
        for node in instance.sources
        if instance.observation.visible is not None
        and not instance.observation.visible[node]
    ]
    check(
        "hidden_nodes deletes nodes from the graph and never a source",
        invisible and all(count > 0 for count in invisible) and not sources_hidden,
        f"{invisible} hidden per cascade, {len(sources_hidden)} sources hidden",
    )

    # The truth is filtered to what remains, or a decoder would be scored on nodes
    # it cannot legally name
    leaked = [
        node
        for instance in hidden
        for node in instance.true_times
        if instance.observation.visible is not None
        and not instance.observation.visible[node]
    ]
    check(
        "a hidden node is removed from the SCORED truth as well",
        not leaked,
        f"{len(leaked)} hidden nodes still in the truth",
    )


def check_bindings(graph: GraphInfo) -> None:
    """@native must raise; the sampling binding must return a real distribution."""
    raised = False
    try:
        unavailable_step_marginals([0], [0])
    except StrategyError as error:
        raised = "@native" in str(error) and "transition kernel" in str(error).lower()

    check(
        "the @native binding RAISES with a message that names the condition",
        raised,
        "arm 3 has no kernel by design",
    )

    environment = MonteCarloEnvironment(graph, "IC", mc_runs=30, base_seed=0)
    frontier = [0, 1, 2]
    marginal = environment.step_marginals(State(sorted(frontier), sorted(frontier)))

    reachable = {
        neighbour for node in frontier for neighbour in graph.out_neighbors(node)
    } - set(frontier)
    outside = [
        node
        for node in range(graph.num_nodes)
        if node not in reachable and node not in frontier and marginal[node] > 0
    ]

    check(
        "the sampling kernel only activates the frontier's own out-neighbours",
        not outside and marginal[sorted(reachable)].sum() > 0,
        f"{len(reachable)} reachable, {len(outside)} impossible activations",
    )

    # The episodes are charged to the arm, which is the cost claim of §2.4.2
    check(
        "a kernel evaluation is charged to the arm's real-episode count",
        environment.episodes_used >= 30,
        f"{environment.episodes_used} episodes for one call at mc_runs=30",
    )


def check_logprob(graph: GraphInfo) -> None:
    """`transition_logprob` has to be the log-likelihood of the marginal it is given."""
    marginal = np.full(graph.num_nodes, 0.25)
    infected = [0, 1]
    gained = [2, 3]

    expected = np.log(0.25) * 2 + np.log(0.75) * (graph.num_nodes - 4)
    actual = transition_logprob(marginal, infected, gained)

    check(
        "transition_logprob is the exact IC factorization of its own marginal",
        abs(actual - expected) < 1e-6,
        f"{actual:.6f} vs {expected:.6f}",
    )


def check_reward_hazards(
    data_dir: Path, graph: GraphInfo, environment: object, task: TaskSpec, instances: list
) -> None:
    """
    The reward is label-free and still not gameable by the easy half.

    A trivial decoder (everyone reachable, parents by BFS) must score badly under
    the likelihood reward, the reported tree score must still expose an under-tree
    decode, and path precision must still be read beside recall.
    """
    both = trivial_decoder_reward(
        instances, graph, task.tree_weight, environment=environment
    )

    class Decoder:
        def reconstruct(self, graph, observation, horizon):
            return reconstruction_algorithms["delayed_bfs"](graph, observation, horizon)

    real, _ = evaluate_reconstructor(
        Decoder(), environment, task, graph, instances, task.tree_weight
    )

    check(
        "a TRIVIAL decoder scores badly under the likelihood reward",
        both["trivial_decoder_reward"] < real.reward,
        f"trivial {both['trivial_decoder_reward']:.4f} against delayed_bfs "
        f"{real.reward:.4f}",
    )

    # The reward never read a label: every per-instance entry is built from the
    # kernel and the observation alone, and the stored history is merged in only
    # by reconstruction_label_metrics afterwards
    entry = real.cost["per_instance"][0]
    check(
        "the in-loop entries carry no label metric",
        not any(key in entry for key in ("path_precision", "event_f1", "tree_score")),
        f"keys: {sorted(entry)}",
    )
    reconstruction_label_metrics(real, instances, task.tree_weight)
    check(
        "the label metrics land only after the search",
        "path_precision" in entry and "tree_score" in real.cost["metrics"],
        f"keys: {sorted(entry)}",
    )

    # The reported tree score still has its own hazard, which is why lambda is
    # printed beside it: Event F1 alone would rank an under-tree decode far higher
    node_only = reconstruction_reward(real.cost["metrics"], 0.0)
    check(
        "the EASY half alone would rank an under-tree decode far higher",
        node_only > real.cost["metrics"]["tree_score"],
        f"event F1 alone {node_only:.4f} against the weighted score "
        f"{real.cost['metrics']['tree_score']:.4f}",
    )

    # The second corner: a decoder that names three TRUE edges and nothing else
    # posts a perfect path precision, and only recall / jaccard say how little
    # of the tree that is
    instance = instances[0]
    named = {node: (0, None) for node in instance.sources}
    for node, causes in instance.true_parents.items():
        if causes and len(named) < len(instance.sources) + 3:
            named[node] = (instance.true_times[node], causes[0])
    metrics = reconstruction_metrics(
        named,
        instance.true_times,
        instance.true_parents,
        instance.num_nodes,
        instance.horizon,
    )
    check(
        "three correct edges score path_precision 1.0 and a LOW recall / jaccard",
        metrics["path_precision"] == 1.0
        and metrics["path_recall"] < 0.5
        and metrics["jaccard"] < 0.5,
        f"precision {metrics['path_precision']:.2f}, recall "
        f"{metrics['path_recall']:.4f}, jaccard {metrics['jaccard']:.4f}",
    )


def check_contract(graph: GraphInfo, instance) -> None:
    """Three incoherent trajectories, three rejections."""
    node = next(
        candidate
        for candidate in range(graph.num_nodes)
        if graph.in_neighbors(candidate)
    )
    non_neighbour = next(
        candidate
        for candidate in range(graph.num_nodes)
        if candidate not in graph.in_neighbors(node) and candidate != node
    )

    cases = {
        "a parent that is not an arc": {node: (1, non_neighbour)},
        "a source at t > 0": {node: (2, None)},
        "a node at t = 0 WITH a parent": {
            node: (0, graph.in_neighbors(node)[0])
        },
        "an empty trajectory": {},
    }

    for label, decoded in cases.items():
        rejected = False
        try:
            validate_reconstruction(decoded, instance, graph)
        except StrategyError:
            rejected = True

        check(f"the contract rejects {label}", rejected)

    # ...a parent that is HIDDEN is rejected, since it is not in the graph at all
    child = next(
        candidate
        for candidate in range(graph.num_nodes)
        if graph.in_neighbors(candidate)
    )
    hidden_parent = graph.in_neighbors(child)[0]
    masked = np.ones(graph.num_nodes, dtype=bool)
    masked[hidden_parent] = False
    hidden_instance = CascadeInstance(
        episode_id="masked",
        graph_id=instance.graph_id,
        num_nodes=instance.num_nodes,
        horizon=instance.horizon,
        sources=list(instance.sources),
        true_times=dict(instance.true_times),
        true_parents=instance.true_parents,
        observation=Observation(
            reported={}, horizon=instance.horizon, num_nodes=instance.num_nodes,
            setting=hidden_nodes, visible=masked,
        ),
        final_state=instance.final_state,
        marginal=instance.marginal,
    )
    rejected = False
    try:
        validate_reconstruction(
            {hidden_parent: (0, None), child: (1, hidden_parent)}, hidden_instance, graph
        )
    except StrategyError:
        rejected = True
    check("the contract rejects a HIDDEN node as parent", rejected)

    # ...and a coherent one is accepted
    parent = graph.in_neighbors(node)[0]
    accepted = validate_reconstruction(
        {parent: (0, None), node: (1, parent)}, instance, graph
    )
    check(
        "a coherent source-plus-child trajectory is accepted",
        accepted == {parent: (0, None), node: (1, parent)},
    )


def check_library(
    data_dir: Path, graph_id: str, graph: GraphInfo, environment: object, task: TaskSpec
) -> None:
    """
    Every member of the condition-1 pool has to return a valid trajectory, on
    EVERY setting: the four masks are four contracts, and a decoder that names a
    hidden node or reads times a setting withholds is a missing row, not a weak one.
    """
    broken = []

    for setting in valid_settings:
        instances = load_cascades(
            str(data_dir),
            "IC",
            "train",
            graph_id=graph_id,
            setting=setting,
            observation_rate=0.35,
            limit=2,
            seed=3,
        )

        for name, decoder in reconstruction_algorithms.items():

            class Anchor:
                canned = True

                def reconstruct(self, graph, observation, horizon, _decoder=decoder):
                    return _decoder(
                        graph,
                        observation,
                        horizon,
                        diffusion_model="IC",
                        predict=getattr(self, "step_marginals", None),
                    )

            try:
                evaluate_reconstructor(
                    Anchor(), environment, task, graph, instances, task.tree_weight
                )
            except Exception as error:
                broken.append(
                    f"{name}@{setting} ({type(error).__name__}: {str(error)[:80]})"
                )

    check(
        "every library decoder returns a valid trajectory on every setting",
        not broken,
        f"{len(reconstruction_algorithms)} decoders x {len(valid_settings)} "
        f"settings, {len(broken)} broken" + (f": {broken}" if broken else ""),
    )


def check_scored_harness(graph: GraphInfo, instance) -> None:
    """Scored mode may override edge_cost and may NOT override reconstruct."""
    allowed = build_strategy(
        "class MyDecoder(ScoredStrategy):\n"
        "    def edge_cost(self, source, target, probability, graph, observation):\n"
        "        return 1.0 / max(float(probability), 1e-9)\n",
        strategy_mode="scored",
    )
    decoded = allowed.reconstruct(graph, instance.observation, instance.horizon)
    check(
        "the scored harness decodes from edge_cost alone",
        bool(validate_reconstruction(decoded, instance, graph)),
        f"{len(decoded)} nodes decoded",
    )

    rejected = False
    try:
        build_strategy(
            "class Cheat(ScoredStrategy):\n"
            "    def reconstruct(self, graph, observation, horizon):\n"
            "        return {}\n",
            strategy_mode="scored",
        )
    except StrategyError:
        rejected = True

    check("scored mode rejects an overridden reconstruct()", rejected)


if __name__ == "__main__":
    work_dir = Path(tempfile.mkdtemp(prefix="gwm-cr-check-"))
    data_dir = work_dir / "data"

    try:
        print(f"[check] generating a tiny traced dataset in {data_dir} ...\n")
        generate(data_dir)

        store = load_graph_store(str(data_dir))
        graph_id = next(iter(store))
        graph = GraphInfo.from_store_entry(store[graph_id])
        print()

        check_traced_dynamics(graph)
        check_parents_are_arcs(data_dir, graph)
        check_settings(data_dir)
        check_bindings(graph)
        check_logprob(graph)

        instances = load_cascades(
            str(data_dir),
            "IC",
            "train",
            graph_id=graph_id,
            setting=partial_times,
            observation_rate=0.35,
            limit=check_instances,
            seed=3,
        )
        task = TaskSpec(
            task="cascade_reconstruction",
            diffusion_model="IC",
            budget=5,
            horizon=check_horizon,
            objective_kind="recover",
            reconstructs=True,
            tree_weight=0.6,
            instances=tuple(instances),
            allowed_ops=("add_node",),
        )
        environment = MonteCarloEnvironment(graph, "IC", mc_runs=10, base_seed=0)

        check_contract(graph, instances[0])
        check_scored_harness(graph, instances[0])
        check_reward_hazards(data_dir, graph, environment, task, instances)
        check_library(data_dir, graph_id, graph, environment, task)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    print(f"\n{'=' * 72}")
    print(f"{len(passed)} passed, {len(failed)} failed")

    if failed:
        print("\nFAILED:")
        for entry in failed:
            print(f"  - {entry}")
        raise SystemExit(1)

    print("cascade-reconstruction contract holds")
