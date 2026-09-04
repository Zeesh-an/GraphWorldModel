import networkx as nx
import numpy as np
import torch

# The four node features GDM's published model was trained on, in the column
# order its checkpoint name spells out (`Fchi_degree_clustering_coefficient_degree_kcore_...`)
gdm_feature_names = ("chi_degree", "clustering_coefficient", "degree", "kcore")

# GDM's published architecture, read off that same checkpoint name: CL20x4, H1x4,
# FL40_30_20_1, concat, negative slope 0.2, dropout 0.3, bias, seed 0
gdm_conv_layers = (20, 20, 20, 20)
gdm_heads = (1, 1, 1, 1)
gdm_fc_layers = (40, 30, 20, 1)
gdm_negative_slope = 0.2
gdm_dropout = 0.3

# The 2019 GATConv kept one projection and one attention vector per layer; PyG 2.x
# splits the attention into a source and a destination half and stores the
# projection under a Linear, whose name moved once across 2.x releases
legacy_projection_names = ("lin.weight", "lin_src.weight", "lin_l.weight")


def gdm_node_features(num_nodes: int, edge_index: np.ndarray) -> np.ndarray:
    """
    `network_dismantling/machine_learning/pytorch/training_data_extractor.py`,
    reproduced without graph_tool: degree over the max degree, chi-square of that
    against its mean, local clustering, core number over the max core number.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from((int(u), int(v)) for u, v in edge_index.T.tolist() if u != v)

    degree = np.asarray(
        [graph.degree(node) for node in range(num_nodes)], dtype=np.float64
    )
    degree = degree / max(float(degree.max()), 1.0)
    mean_degree = float(degree.mean())
    chi_degree = (
        (degree - mean_degree) ** 2 / mean_degree
        if mean_degree > 0
        else np.zeros_like(degree)
    )

    clustering = nx.clustering(graph)
    clustering_coefficient = np.asarray([clustering[node] for node in range(num_nodes)])

    core = nx.core_number(graph)
    kcore = np.asarray([core[node] for node in range(num_nodes)], dtype=np.float64)
    kcore = kcore / max(float(kcore.max()), 1.0)

    return np.column_stack(
        [chi_degree, clustering_coefficient, degree, kcore]
    )  # shape: (N, 4)


def convert_legacy_gat_state(legacy: dict, target_keys) -> dict:
    """
    Map a PyG 1.x GATConv state dict onto the installed GATConv's parameter names.

    1.x stored `weight` as (in, heads * out) and applied it as `x @ weight`; 2.x
    stores a Linear whose weight is (heads * out, in), so the tensor is transposed.
    1.x scored an edge as `cat(x_i, x_j) . att` with `x_i` the target node, so the
    first half of `att` is 2.x's `att_dst` and the second half its `att_src`.
    Everything outside the convolutions (the per-layer Linear and the regressor)
    already shares its names.
    """
    target_keys = set(target_keys)
    converted = {}

    for key, value in legacy.items():
        prefix, _, leaf = key.rpartition(".")

        if prefix.startswith("convolutional_layers") and leaf == "weight":
            candidates = [f"{prefix}.{name}" for name in legacy_projection_names]
            present = [name for name in candidates if name in target_keys]
            if not present:
                raise KeyError(
                    f"{key}: the installed GATConv has none of {candidates}; "
                    f"its keys are {sorted(k for k in target_keys if k.startswith(prefix))}"
                )
            converted[present[0]] = value.t().contiguous()
        elif prefix.startswith("convolutional_layers") and leaf == "att":
            out_channels = value.shape[-1] // 2
            converted[f"{prefix}.att_dst"] = value[..., :out_channels].contiguous()
            converted[f"{prefix}.att_src"] = value[..., out_channels:].contiguous()
        else:
            converted[key] = value

    missing = target_keys - set(converted)
    unexpected = set(converted) - target_keys
    if missing or unexpected:
        raise KeyError(
            f"legacy GAT checkpoint does not fit the installed model: "
            f"missing {sorted(missing)}, unexpected {sorted(unexpected)}"
        )

    return converted


def undirected_edge_index(edge_index: np.ndarray) -> torch.Tensor:
    """Both arc directions, deduplicated, which is what GDM's `prepare_graph` feeds the GAT."""
    pairs = {(int(u), int(v)) for u, v in edge_index.T.tolist() if u != v}
    pairs |= {(v, u) for u, v in pairs}
    ordered = sorted(pairs)

    return torch.tensor(ordered, dtype=torch.long).t().contiguous()  # shape: (2, 2E)
