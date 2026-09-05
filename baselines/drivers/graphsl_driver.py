"""
Driver for the GraphSL package: our episodes in, one source set per episode out.

Runs inside GraphSL's OWN virtualenv as a subprocess, so nothing here may import
from the rest of this repo. It reads two npz files written by
`registry._graphsl_export` and writes `predictions.json`.

Six published methods behind one package (JOSS 9(99):6796, 2024):
LPSI, NetSleuth, OJC, GCNSI, IVGD and SL-VAE. `GWM_GRAPHSL_METHOD` picks one.

THREE THINGS THIS DELIBERATELY DOES NOT DO, each of which would invalidate the
comparison it exists to produce:

  * **It never calls GraphSL's `test()` / `infer()`.** Those score against the
    label column internally and return only an aggregate `Metric`, so no
    per-instance prediction escapes them. We need the SOURCE SET itself, because
    our own referee scores it: the same discipline every external baseline in
    this repo follows. The prediction lines are therefore reproduced here from
    each method's own `test` body.

  * **It never lets a label reach an evaluation-split prediction.** `train()`
    legitimately sees labels: every one of these methods tunes a threshold or
    fits weights on labelled data, exactly as our outer loop selects a program on
    labelled episodes. So `train` is given the SELECTION split only, and the
    prediction pass is label-free for all six.

  * **It does not use GraphSL's thresholding to form the source set.** GraphSL
    cuts its score vector at a tuned threshold, which yields a variable-size set;
    our table compares every arm at matched `k` (the instance's own source count,
    which is the published given-k convention). So the set here is the TOP-K of
    the same score vector. The threshold GraphSL chose is reported alongside for
    anyone who wants its own convention back.

Environment:
    GWM_GRAPHSL_METHOD   lpsi | netsleuth | ojc | gcnsi | ivgd | slvae
    GWM_GRAPHSL_EPOCHS   training epochs for the three learned methods
    GWM_GRAPHSL_SEED     random seed
"""

# `GraphSL` is installed in baselines/external/graphsl/.venv, NOT in ours: that
# isolation is the whole point of the external-baseline design, since its pins
# and ours cannot coexist. An editor resolving this file against the project
# interpreter therefore reports every GraphSL import as missing, and it is right
# to: this file never runs under that interpreter. Scoped to the file rather than
# set in .vscode/settings.json so it travels with the code and does not silence a
# genuine missing import anywhere else.
# pyright: reportMissingImports=false

import json
import os
import sys
import numpy as np
import scipy.sparse as sp
import torch

methods = ("lpsi", "netsleuth", "ojc", "gcnsi", "ivgd", "slvae")


def load_inputs(work_dir):
    graph = np.load(os.path.join(work_dir, "graph.npz"))
    adjacency = sp.csr_matrix(
        (graph["data"], graph["indices"], graph["indptr"]),
        shape=tuple(graph["shape"]),
    )

    payload = np.load(os.path.join(work_dir, "instances.npz"), allow_pickle=True)
    episodes = [str(name) for name in payload["episode_ids"]]

    return adjacency, episodes, payload


def build_dataset(payload, mask):
    """
    GraphSL's `influ_mat` layout: (N, 2), column 0 the seed vector, column -1 the
    diffusion vector. Every method indexes exactly those two columns, so a
    two-column stack is the whole contract.
    """
    seeds, observations = payload["seeds"], payload["observations"]

    return [
        torch.tensor(
            np.stack([seeds[index], observations[index]], axis=1), dtype=torch.float32
        )
        for index in np.flatnonzero(mask)
    ]


def top_k(scores, budget):
    scores = np.asarray(scores, dtype=np.float64).ravel()
    # Ties broken by node id so a rerun is reproducible
    order = np.lexsort((np.arange(scores.size), -scores))

    return [int(node) for node in order[: max(1, int(budget))]]


def predict_lpsi(adjacency, train_dataset, all_datasets, _payload, _epochs, _seed):
    from GraphSL.Prescribed import LPSI
    from scipy.sparse.csgraph import laplacian as csgraph_laplacian

    model = LPSI()
    alpha, threshold, _, _, _ = model.train(adjacency, train_dataset)

    laplacian = torch.tensor(
        csgraph_laplacian(adjacency, normed=True).toarray(),
        dtype=torch.float32,
        device=model.device,
    )
    num_node = adjacency.shape[0]

    scores = [
        model.predict(
            laplacian,
            num_node,
            alpha,
            torch.tensor(mat[:, -1], dtype=torch.float32, device=model.device),
        )
        .cpu()
        .detach()
        .numpy()
        for mat in all_datasets
    ]

    return scores, {"alpha": float(alpha), "threshold": float(threshold)}


def predict_netsleuth(adjacency, train_dataset, all_datasets, _payload, _epochs, _seed):
    import networkx as nx
    from GraphSL.Prescribed import NetSleuth

    model = NetSleuth()
    opt_k, _, _ = model.train(adjacency, train_dataset)
    graph = nx.from_scipy_sparse_array(adjacency)

    # NetSleuth returns a BINARY seed vector, not a ranking, so its top-k is a
    # tie among the nodes it named. The observation is added as a tiny tie-break
    # so the k it selected survives and the rest fall in a stable order.
    scores = []
    for mat in all_datasets:
        diff = mat[:, -1].numpy()
        picked = model.predict(graph, opt_k, torch.tensor(diff)).cpu().numpy()
        scores.append(np.asarray(picked, dtype=np.float64) + 1e-6 * diff)

    return scores, {"k": int(opt_k)}


def predict_ojc(adjacency, train_dataset, all_datasets, _payload, _epochs, _seed):
    import networkx as nx
    from GraphSL.Prescribed import OJC

    model = OJC()
    opt_y, _, _ = model.train(adjacency, train_dataset)
    graph = nx.from_scipy_sparse_array(adjacency)

    scores = []
    for mat in all_datasets:
        # Argument shapes taken from OJC.test verbatim: `target` is the diffusion
        # vector as a TENSOR (predict calls torch.zeros_like on it), `I` is the
        # infected index list, and `num_source` is the infected count
        influ_vec = mat[:, -1].to(model.device)
        infected = torch.where(influ_vec == 1)[0].tolist()
        num_source = int(torch.sum(influ_vec == 1).item()) or 1

        picked = (
            model.predict(graph, opt_y, infected, influ_vec, num_source).cpu().numpy()
        )
        diff = influ_vec.cpu().numpy()
        scores.append(np.asarray(picked, dtype=np.float64).ravel() + 1e-6 * diff)

    return scores, {"Y": int(opt_y)}


def _gcnsi_features(adjacency, dataset, alpha=0.01):
    """
    GCNSI's four-channel LPSI input, rebuilt exactly as its own `test` does.

    Reproduced rather than imported because GraphSL computes it inline inside a
    method that returns only aggregate metrics.
    """
    from GraphSL.Prescribed import LPSI
    from scipy.sparse.csgraph import laplacian as csgraph_laplacian

    lpsi = LPSI()
    num_node = adjacency.shape[0]
    laplacian = csgraph_laplacian(adjacency, normed=True).toarray()

    features = []
    for mat in dataset:
        diff = mat[:, -1].numpy()
        v3 = diff.copy()
        v4 = diff.copy()
        v3[v3 == 0] = -1
        v4[v4 == 0] = -1
        # GCNSI's paper stacks the raw state, its sign flip, and two LPSI fields
        channels = [
            torch.tensor(diff, dtype=torch.float).unsqueeze(dim=1),
            torch.tensor(v3, dtype=torch.float).unsqueeze(dim=1),
            torch.tensor(
                lpsi.predict(
                    torch.tensor(laplacian, dtype=torch.float32, device=lpsi.device),
                    num_node,
                    alpha,
                    torch.tensor(v3, dtype=torch.float32, device=lpsi.device),
                ).cpu(),
                dtype=torch.float,
            ).unsqueeze(dim=1),
            torch.tensor(
                lpsi.predict(
                    torch.tensor(laplacian, dtype=torch.float32, device=lpsi.device),
                    num_node,
                    alpha,
                    torch.tensor(v4, dtype=torch.float32, device=lpsi.device),
                ).cpu(),
                dtype=torch.float,
            ).unsqueeze(dim=1),
        ]
        features.append(torch.cat(channels, dim=1))

    return features


def predict_gcnsi(adjacency, train_dataset, all_datasets, _payload, epochs, seed):
    from GraphSL.GNN.GCNSI.main import GCNSI

    model = GCNSI()
    gcnsi_model, threshold, _, _, _ = model.train(
        adjacency, train_dataset, num_epoch=epochs, random_seed=seed
    )

    coo = adjacency.tocoo()
    edge_index = torch.tensor(
        np.vstack([coo.row, coo.col]), dtype=torch.long, device=model.device
    )
    gcnsi_model.eval()

    scores = []
    with torch.no_grad():
        for features in _gcnsi_features(adjacency, all_datasets):
            prediction = gcnsi_model(features.to(model.device), edge_index)
            scores.append(torch.softmax(prediction, dim=1)[:, 1].cpu().numpy())

    return scores, {"threshold": float(threshold)}


def predict_ivgd(adjacency, train_dataset, all_datasets, _payload, epochs, seed):
    from GraphSL.GNN.IVGD.main import IVGD

    model = IVGD()
    diffusion = model.train_diffusion(
        adjacency, train_dataset, num_epoch=max(10, epochs // 4), random_seed=seed
    )
    ivgd_model, threshold, _, _, _ = model.train(
        adjacency, train_dataset, diffusion, num_epoch=epochs, random_seed=seed
    )

    # Argument shapes taken from IVGD.test verbatim. The two stages are the
    # paper's own: run the pretrained diffusion GNN BACKWARDS to get a raw seed
    # estimate, then pass it through the validity-aware correction network. That
    # correction is the unrolled per-instance optimization which keeps IVGD in
    # the non-amortized row of the taxonomy.
    ivgd_model = ivgd_model.to(model.device)
    lamda = 1e-3

    scores = []
    with torch.no_grad():
        for mat in all_datasets:
            influ_vec = mat[:, -1].unsqueeze(dim=-1).to(model.device)
            seed_preds = diffusion.backward(adjacency, influ_vec).to(model.device)

            correction = ivgd_model(seed_preds, seed_preds, lamda)
            correction = torch.softmax(correction, dim=1)[:, 1]
            scores.append(correction.cpu().numpy().ravel())

    return scores, {"threshold": float(threshold)}


def predict_slvae(adjacency, train_dataset, all_datasets, _payload, epochs, seed):
    from GraphSL.GNN.SLVAE.main import SLVAE
    from torch.optim import Adam

    model = SLVAE()
    slvae_model, seed_vae_train, threshold, _, _, _ = model.train(
        adjacency, train_dataset, num_epoch=epochs, random_seed=seed
    )

    # SL-VAE's inference is itself an optimization: initialize a seed vector from
    # the VAE's training latents, then descend the forward-reconstruction loss on
    # it per instance. Reproduced from `SLVAE.infer`, minus its metric block. Note
    # what this costs: one gradient loop PER TEST INSTANCE is exactly the
    # per-instance optimization research/source_localization.md §1 puts SL-VAE in
    # the non-amortized row for.
    slvae_model = slvae_model.to(model.device)
    slvae_model.eval()
    for parameter in slvae_model.parameters():
        parameter.requires_grad = False

    seed_mean = torch.mean(seed_vae_train, dim=0).unsqueeze(dim=-1).to(model.device)
    seed_infer = []
    for _ in range(len(all_datasets)):
        seed_hat, _, _, _ = slvae_model(seed_mean, False)
        seed_infer.append(seed_hat)

    for seed_vector in seed_infer:
        seed_vector.requires_grad = True

    optimizer = Adam(seed_infer, lr=1e-4)
    for _ in range(max(10, epochs // 10)):
        for index, mat in enumerate(all_datasets):
            influ_vec = mat[:, -1].unsqueeze(dim=-1).float().to(model.device)
            optimizer.zero_grad()
            seed_hat, _, _, influ_hat = slvae_model(seed_infer[index], False)
            loss = slvae_model.infer_loss(
                influ_vec, influ_hat, seed_hat, seed_vae_train
            )
            loss.backward()
            optimizer.step()

    scores = [
        vector.cpu().detach().numpy().ravel() for vector in seed_infer
    ]

    return scores, {"threshold": float(threshold)}


predictors = {
    "lpsi": predict_lpsi,
    "netsleuth": predict_netsleuth,
    "ojc": predict_ojc,
    "gcnsi": predict_gcnsi,
    "ivgd": predict_ivgd,
    "slvae": predict_slvae,
}


if __name__ == "__main__":
    work_dir = sys.argv[1]
    method = os.environ.get("GWM_GRAPHSL_METHOD", "lpsi").lower()
    epochs = int(os.environ.get("GWM_GRAPHSL_EPOCHS", "50"))
    seed = int(os.environ.get("GWM_GRAPHSL_SEED", "0"))

    if method not in predictors:
        raise SystemExit(
            f"unknown GWM_GRAPHSL_METHOD {method!r}; choose one of {methods}"
        )

    adjacency, episodes, payload = load_inputs(work_dir)
    is_train = payload["is_train"].astype(bool)
    budgets = payload["budgets"].astype(int)

    train_dataset = build_dataset(payload, is_train)
    all_datasets = build_dataset(payload, np.ones(len(episodes), dtype=bool))

    print(
        f"[graphsl] method={method} nodes={adjacency.shape[0]} "
        f"train={len(train_dataset)} predict={len(all_datasets)} epochs={epochs}",
        flush=True,
    )

    scores, info = predictors[method](
        adjacency, train_dataset, all_datasets, payload, epochs, seed
    )

    predictions = {
        episode: top_k(score, budget)
        for episode, score, budget in zip(episodes, scores, budgets)
    }

    with open(os.path.join(work_dir, "predictions.json"), "w") as handle:
        json.dump({"method": method, "info": info, "sources": predictions}, handle)

    print(f"[graphsl] wrote {len(predictions)} predictions ({info})", flush=True)
