"""
Build the `.SG` training file DeepIM needs, from OUR graphs.

DeepIM was blocked because its repo ships an empty `data/` folder. Reading
`genim.py` shows the file is not exotic: it is a pickled dict with two keys:

    graph = pickle.load(open("data/<dataset>_mean_<DM><10*rate>.SG", "rb"))
    adj, inverse_pairs = graph["adj"], graph["inverse_pairs"]

    adj            scipy sparse (N, N) adjacency
    inverse_pairs  torch tensor (n_samples, N, 2)
                   [..., 0] = seed indicator, [..., 1] = influenced indicator

That second tensor is exactly what our simulator already produces: pick a seed
set, diffuse to termination, record who ended up infected. So DeepIM is
unblockable: it needs data we can generate, not data only the authors have.

    python -m baselines.deepim_data --data-dir results/ba40/data \
        --dataset jazz --diffusion-model LT --seed-rate 1 --samples 1000
"""

import argparse
import os
import pickle
from pathlib import Path
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm

from baselines.registry import external_baselines
from coding_agent.types import GraphInfo
from data.wm_simulator import ActionOp, Simulator, State
from world_model.wm_data import load_graph_store

# DeepIM sweeps these four budgets; the filename encodes 10x the percentage
seed_rates = (1, 5, 10, 20)
default_samples = 1000


def _adjacency(graph: GraphInfo) -> sp.csr_matrix:
    rows = graph.edge_index[0]
    columns = graph.edge_index[1]
    values = np.ones(rows.shape[0], dtype=np.float32)

    return sp.csr_matrix(
        (values, (rows, columns)), shape=(graph.num_nodes, graph.num_nodes)
    )


def build_inverse_pairs(
    graph: GraphInfo,
    bundle_graph,
    ic_prob_map: dict,
    diffusion_model: str,
    budget: int,
    samples: int,
    seed: int = 42,
) -> torch.Tensor:
    """
    One (seed vector, influenced vector) row per simulated cascade.

    The seed sets are drawn from a MIXTURE of samplers, not uniformly at
    random. DeepIM autoencodes seed vectors and then optimises in that latent
    space, so the reachable solutions are only as good as the sets the
    autoencoder was trained on. Train it on uniform-random 16-subsets and the
    manifold contains nothing but random-quality answers: the optimisation
    converges to seeds that score at or below the random baseline, which is
    exactly what we measured. Degree- and PageRank-biased draws put
    high-influence sets inside the manifold; the uniform third keeps the
    forward model honest about what a bad set looks like.
    """
    rng = np.random.default_rng(seed)
    pairs = np.zeros((samples, graph.num_nodes, 2), dtype=np.float32)

    degrees = np.array(
        [float(graph.degree(node)) for node in range(graph.num_nodes)]
    )
    view = nx.Graph()
    view.add_nodes_from(range(graph.num_nodes))
    view.add_edges_from(zip(graph.edge_index[0], graph.edge_index[1], strict=True))
    ranks = nx.pagerank(view)
    pagerank = np.array([ranks.get(node, 0.0) for node in range(graph.num_nodes)])

    def normalize(weights):
        total = weights.sum()

        return weights / total if total > 0 else None

    samplers = [None, normalize(degrees), normalize(pagerank)]

    for index in tqdm(range(samples), desc=f"deepim pairs k={budget}"):
        weights = samplers[index % len(samplers)]
        # A biased draw needs at least `budget` nodes with non-zero weight
        if weights is not None and int((weights > 0).sum()) < budget:
            weights = None

        seeds = rng.choice(
            graph.num_nodes, size=budget, replace=False, p=weights
        )

        simulator = Simulator(
            bundle_graph, ic_prob_map=ic_prob_map, seed=int(rng.integers(0, 2**31 - 1))
        )
        simulator.reset(diffusion_model)

        state = State(infected=[], frontier=[])
        action = [ActionOp("add_node", int(node)) for node in seeds]

        # Diffuse to termination: DeepIM's target is the FINAL infected set
        for _ in range(graph.num_nodes):
            state, _, _ = simulator.advance_marginal(action, 1)
            action = []
            if not state.frontier:
                break

        pairs[index, seeds, 0] = 1.0
        pairs[index, list(state.infected), 1] = 1.0

    return torch.from_numpy(pairs)


def _bundle_from_graph(graph: GraphInfo):
    """
    Rebuild the networkx view + edge-probability map the simulator needs.

    Used when the caller already holds a GraphInfo (the pipeline adapter path),
    so we do not re-derive the graph from a dataset name that may not match any
    generator family.
    """
    import networkx as nx

    view = nx.DiGraph() if graph.directed else nx.Graph()
    view.add_nodes_from(range(graph.num_nodes))
    probability_map = {}

    for column in range(graph.edge_index.shape[1]):
        source = int(graph.edge_index[0, column])
        target = int(graph.edge_index[1, column])
        view.add_edge(source, target)
        probability_map[(source, target)] = float(graph.ic_probs[column])

    return view, probability_map


def build_sg_file(
    data_dir: str,
    dataset: str,
    diffusion_model: str,
    seed_rate: int,
    samples: int,
    graph_id: str | None = None,
    out_dir: Path | None = None,
    seed: int = 42,
    graph: GraphInfo | None = None,
) -> Path:
    if graph is not None:
        nx_graph, ic_prob_map = _bundle_from_graph(graph)
    else:
        from data.wm_graphs import make_real_bundle, make_synthetic_bundle
        from data.generate_wm_data import synthetic_families

        store = load_graph_store(data_dir)
        key = graph_id or next(iter(store))
        graph = GraphInfo.from_store_entry(store[key])

        # The simulator needs the networkx bundle, not just the tensors
        if dataset in synthetic_families:
            bundle = make_synthetic_bundle(
                dataset, index=0, num_nodes=graph.num_nodes, seed=seed
            )
        else:
            bundle = make_real_bundle(dataset)

        nx_graph, ic_prob_map = bundle.nx_graph, bundle.ic_prob_map

    budget = max(1, round(graph.num_nodes * seed_rate / 100))
    pairs = build_inverse_pairs(
        graph,
        nx_graph,
        ic_prob_map,
        diffusion_model,
        budget,
        samples,
        seed=seed,
    )

    out_dir = Path(out_dir or external_baselines["deepim"].directory / "data")
    os.makedirs(out_dir, exist_ok=True)
    path = out_dir / f"{dataset}_mean_{diffusion_model}{10 * seed_rate}.SG"

    with open(path, "wb") as handle:
        pickle.dump({"adj": _adjacency(graph), "inverse_pairs": pairs}, handle)

    print(
        f"[deepim] wrote {path}: adj {graph.num_nodes}x{graph.num_nodes}, "
        f"inverse_pairs {tuple(pairs.shape)}, k={budget} ({seed_rate}% of N)"
    )

    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate DeepIM .SG training files from our graphs"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="graph store directory, e.g. results/jazz/data (default: required).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="dataset name; becomes DeepIM's -d argument (default: required).",
    )
    parser.add_argument(
        "--graph-id",
        type=str,
        default=None,
        help="graph id inside the store (default: the first).",
    )
    parser.add_argument(
        "--diffusion-model",
        type=str,
        default="LT",
        choices=["IC", "LT"],
        help="diffusion model DeepIM will be trained for (default: LT).",
    )
    parser.add_argument(
        "--seed-rate",
        type=int,
        nargs="+",
        default=list(seed_rates),
        choices=list(seed_rates),
        help=f"budget percentages to build files for (default: {' '.join(map(str, seed_rates))}).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=default_samples,
        help=f"cascades simulated per file (default: {default_samples}).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="where to write the .SG files (default: baselines/external/deepim/data).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)."
    )

    args = parser.parse_args()

    for rate in args.seed_rate:
        build_sg_file(
            args.data_dir,
            args.dataset,
            args.diffusion_model,
            rate,
            args.samples,
            graph_id=args.graph_id,
            out_dir=Path(args.out_dir) if args.out_dir else None,
            seed=args.seed,
        )
