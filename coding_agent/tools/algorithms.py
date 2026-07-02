"""
Named classical IM algorithms, composed from primitives.

Every algorithm has the signature (g, budget, diffusion_model, **kw) -> list[int]
and returns a seed set of size `budget`. These are the callable surface offered to
the coding agent; it may call one directly or compose several across timesteps.
"""

import heapq
import numpy as np

from coding_agent.types import GraphInfo
from coding_agent.tools import primitives as P


def _top_k_by_score(score: np.ndarray, k: int) -> list[int]:
    return [int(v) for v in np.argsort(-score)[:k]]


def high_degree(
    g: GraphInfo, budget: int, diffusion_model: str = "IC", **_
) -> list[int]:
    """Top-k highest total-degree nodes (Degree heuristic)."""
    return P.get_top_degree_nodes(g, budget)


def weighted_degree(
    g: GraphInfo, budget: int, diffusion_model: str = "IC", **_
) -> list[int]:
    """Top-k by summed outgoing IC transmission probability."""
    return _top_k_by_score(P.compute_weighted_degree(g), budget)


def degree_discount(
    g: GraphInfo, budget: int, diffusion_model: str = "IC", **_
) -> list[int]:
    """DegreeDiscount (Chen et al. 2009): discount a node's degree for already-chosen neighbors."""
    deg = P.compute_degree(g).copy()
    chosen: list[int] = []
    discounted = deg.copy()
    for _ in range(budget):
        v = int(
            np.argmax(
                [discounted[i] if i not in chosen else -1 for i in range(g.num_nodes)]
            )
        )
        chosen.append(v)
        for w in g.out_neighbors(v) + g.in_neighbors(v):
            discounted[w] -= 1
    return chosen


def pagerank_seeds(
    g: GraphInfo, budget: int, diffusion_model: str = "IC", **_
) -> list[int]:
    """Top-k PageRank nodes."""
    return _top_k_by_score(P.compute_pagerank(g), budget)


def vanilla_greedy(
    g: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    mc_runs: int = 30,
    horizon: int = 20,
    **_,
) -> list[int]:
    """Greedy marginal-gain (Kempe et al. 2003)."""
    seeds: list[int] = []
    for _ in range(budget):
        best_v, best_gain = -1, float("-inf")
        for v in range(g.num_nodes):
            if v in seeds:
                continue
            gain = P.compute_marginal_gain(
                g, seeds, v, diffusion_model, mc_runs, horizon
            )
            if gain > best_gain:
                best_gain, best_v = gain, v
        seeds.append(best_v)
    return seeds


def celf(
    g: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    mc_runs: int = 30,
    horizon: int = 20,
    **_,
) -> list[int]:
    """CELF (Leskovec et al. 2007): lazy-forward greedy with a max-heap of marginal gains."""
    seeds: list[int] = []
    heap: list[tuple[float, int, int]] = []  # (-gain, node, last_updated_round)
    for v in range(g.num_nodes):
        gain = P.mc_simulate_spread(g, [v], diffusion_model, mc_runs, horizon)
        heapq.heappush(heap, (-gain, v, 0))
    for rnd in range(1, budget + 1):
        while True:
            neg_gain, v, last = heapq.heappop(heap)
            if last == rnd:
                seeds.append(v)
                break
            fresh = P.mc_simulate_spread(
                g, seeds + [v], diffusion_model, mc_runs, horizon
            ) - P.mc_simulate_spread(g, seeds, diffusion_model, mc_runs, horizon)
            heapq.heappush(heap, (-fresh, v, rnd))
    return seeds


def ris_basic(
    g: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    theta: int = 2000,
    **_,
) -> list[int]:
    """Basic Reverse Influence Sampling (Borgs et al. 2014). IC only."""
    rr = P.batch_reverse_sample(g, theta=theta, seed=0)
    seeds = P.ris_select(rr, budget, g.num_nodes)
    # pad with degree if coverage ran out
    if len(seeds) < budget:
        for v in P.get_top_degree_nodes(g, g.num_nodes):
            if v not in seeds:
                seeds.append(v)
            if len(seeds) == budget:
                break
    return seeds


def community_im(
    g: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    **_,
) -> list[int]:
    """Community-IM: split budget across label-propagation communities, degree within each."""
    comm = P.detect_communities(g)
    alloc = P.allocate_budget(comm, budget)
    members: dict[int, list[int]] = {}
    for v, c in comm.items():
        members.setdefault(c, []).append(v)
    deg = P.compute_degree(g)
    seeds: list[int] = []
    for c, b in alloc.items():
        ranked = sorted(members[c], key=lambda v: deg[v], reverse=True)
        seeds.extend(ranked[:b])
    return seeds[:budget]


def pagerank_greedy(
    g: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    pool: int = 30,
    mc_runs: int = 30,
    horizon: int = 20,
    **_,
) -> list[int]:
    """Hybrid: narrow to top PageRank candidates, then greedy marginal-gain over them."""
    candidates = _top_k_by_score(P.compute_pagerank(g), min(pool, g.num_nodes))
    seeds: list[int] = []
    for _ in range(budget):
        best_v, best_gain = -1, float("-inf")
        for v in candidates:
            if v in seeds:
                continue
            gain = P.compute_marginal_gain(
                g, seeds, v, diffusion_model, mc_runs, horizon
            )
            if gain > best_gain:
                best_gain, best_v = gain, v
        if best_v < 0:
            break
        seeds.append(best_v)
    return _pad_seeds(seeds, g, budget)


def hill_climbing(
    g: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    mc_runs: int = 30,
    horizon: int = 20,
    rounds: int = 3,
    **_,
) -> list[int]:
    """Local search: start from degree, 1-swap while spread improves."""
    seeds = P.get_top_degree_nodes(g, budget)
    best = P.mc_simulate_spread(g, seeds, diffusion_model, mc_runs, horizon)
    for _ in range(rounds):
        improved = False
        for i in range(len(seeds)):
            for v in range(g.num_nodes):
                if v in seeds:
                    continue
                trial = list(seeds)
                trial[i] = v
                sp = P.mc_simulate_spread(g, trial, diffusion_model, mc_runs, horizon)
                if sp > best:
                    best, seeds, improved = sp, trial, True
                    break
            if improved:
                break
        if not improved:
            break
    return seeds


# --- shared selection helpers (used by several families) --------------------
def _pad_seeds(seeds: list[int], g: GraphInfo, budget: int) -> list[int]:
    """Dedupe + pad to budget with high-degree fillers."""
    out = list(dict.fromkeys(int(s) for s in seeds))
    if len(out) < budget:
        for v in P.get_top_degree_nodes(g, g.num_nodes):
            if v not in out:
                out.append(v)
            if len(out) == budget:
                break
    return out[:budget]


def _greedy_discount_select(g: GraphInfo, scores: np.ndarray, budget: int) -> list[int]:
    """Pick top-k by score, halving an out-neighbor's score when its source is chosen."""
    s = scores.astype(float).copy()
    chosen: list[int] = []
    for _ in range(budget):
        v = int(
            np.argmax([s[i] if i not in chosen else -1e18 for i in range(g.num_nodes)])
        )
        chosen.append(v)
        for w in g.out_neighbors(v):
            s[w] *= 0.5
    return chosen


# --- Greedy family (additions) ---------------------------------------------
def celf_pp(
    g: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    mc_runs: int = 30,
    horizon: int = 20,
    **_,
) -> list[int]:
    """CELF++ (Goyal et al. 2011): CELF with a secondary cache (simplified; selection = CELF)."""
    seeds: list[int] = []
    heap: list[tuple[float, int, int]] = []
    for v in range(g.num_nodes):
        gain = P.mc_simulate_spread(g, [v], diffusion_model, mc_runs, horizon)
        heapq.heappush(heap, (-gain, v, 0))
    for rnd in range(1, budget + 1):
        while True:
            neg_gain, v, last = heapq.heappop(heap)
            if last == rnd:
                seeds.append(v)
                break
            fresh = P.mc_simulate_spread(
                g, seeds + [v], diffusion_model, mc_runs, horizon
            ) - P.mc_simulate_spread(g, seeds, diffusion_model, mc_runs, horizon)
            heapq.heappush(heap, (-fresh, v, rnd))
    return seeds


def adaptive_greedy(
    g: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    base_mc: int = 10,
    refine_mc: int = 40,
    horizon: int = 20,
    **_,
) -> list[int]:
    """Adaptive Greedy: coarse MC to rank, refined MC on the top few (adaptive sample count)."""
    seeds: list[int] = []
    for _ in range(budget):
        coarse = sorted(
            (
                (
                    P.compute_marginal_gain(
                        g, seeds, v, diffusion_model, base_mc, horizon
                    ),
                    v,
                )
                for v in range(g.num_nodes)
                if v not in seeds
            ),
            reverse=True,
        )
        best_v, best_gain = coarse[0][1], float("-inf")
        for _, v in coarse[:5]:
            gain = P.compute_marginal_gain(
                g, seeds, v, diffusion_model, refine_mc, horizon
            )
            if gain > best_gain:
                best_gain, best_v = gain, v
        seeds.append(best_v)
    return seeds


# --- Centrality (additions) -------------------------------------------------
def eigenvector_seeds(
    g: GraphInfo, budget: int, diffusion_model: str = "IC", **_
) -> list[int]:
    """Top-k eigenvector-centrality nodes."""
    return _top_k_by_score(P.compute_centrality(g, "eigenvector"), budget)


def closeness_seeds(
    g: GraphInfo, budget: int, diffusion_model: str = "IC", **_
) -> list[int]:
    """Top-k closeness-centrality nodes."""
    return _top_k_by_score(P.compute_centrality(g, "closeness"), budget)


# --- RIS family (additions; IC) --------------------------------------------
def tim(
    g: GraphInfo, budget: int, diffusion_model: str = "IC", epsilon: float = 0.2, **_
) -> list[int]:
    """TIM/TIM+ (Tang et al. 2014): RIS with an estimated sample size."""
    theta = P.estimate_sample_size(g, budget, epsilon=epsilon)
    rr = P.batch_reverse_sample(g, theta=theta, seed=0)
    return _pad_seeds(P.ris_select(rr, budget, g.num_nodes), g, budget)


def imm(
    g: GraphInfo, budget: int, diffusion_model: str = "IC", epsilon: float = 0.2, **_
) -> list[int]:
    """IMM (Tang et al. 2015): two-phase sample-size refinement over RIS (simplified)."""
    theta0 = P.estimate_sample_size(g, budget, epsilon=epsilon * 2)
    rr = P.batch_reverse_sample(g, theta=theta0, seed=0)
    theta = max(theta0, P.estimate_sample_size(g, budget, epsilon=epsilon))
    if theta > theta0:
        rr = rr + P.batch_reverse_sample(g, theta=theta - theta0, seed=1)
    return _pad_seeds(P.ris_select(rr, budget, g.num_nodes), g, budget)


def ssa(g: GraphInfo, budget: int, diffusion_model: str = "IC", **_) -> list[int]:
    """SSA/D-SSA (Nguyen et al. 2016): doubling RIS until the top-k stabilizes (simplified)."""
    theta = 500
    rr = P.batch_reverse_sample(g, theta=theta, seed=0)
    prev = set(P.ris_select(rr, budget, g.num_nodes))
    for it in range(1, 5):
        theta *= 2
        rr = rr + P.batch_reverse_sample(g, theta=theta, seed=it)
        cur = set(P.ris_select(rr, budget, g.num_nodes))
        if len(cur & prev) >= max(1, int(0.9 * budget)):
            prev = cur
            break
        prev = cur
    return _pad_seeds(sorted(prev), g, budget)


def filtered_ris(
    g: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    theta: int = 2000,
    min_size: int = 2,
    **_,
) -> list[int]:
    """Filtered RIS: drop tiny RR sets before coverage selection."""
    rr = [
        s for s in P.batch_reverse_sample(g, theta=theta, seed=0) if len(s) >= min_size
    ]
    if not rr:
        rr = P.batch_reverse_sample(g, theta=theta, seed=0)
    return _pad_seeds(P.ris_select(rr, budget, g.num_nodes), g, budget)


# --- Path-based (simplified via truncated path-sum) ------------------------
def sp1m(
    g: GraphInfo, budget: int, diffusion_model: str = "IC", max_hops: int = 3, **_
) -> list[int]:
    """SP1M/SPM (Kimura & Saito 2006): top-k by shortest-path influence (truncated path-sum)."""
    return _top_k_by_score(P.path_influence_scores(g, max_hops=max_hops), budget)


def mia_pmia(
    g: GraphInfo, budget: int, diffusion_model: str = "IC", max_hops: int = 2, **_
) -> list[int]:
    """MIA/PMIA (Chen et al. 2010): local-influence-tree score (simplified), discounted selection."""
    return _greedy_discount_select(
        g, P.path_influence_scores(g, max_hops=max_hops), budget
    )


def ldag(
    g: GraphInfo, budget: int, diffusion_model: str = "IC", max_hops: int = 3, **_
) -> list[int]:
    """LDAG (Chen et al. 2010): local-DAG influence (simplified deeper path-sum), discounted selection."""
    return _greedy_discount_select(
        g, P.path_influence_scores(g, max_hops=max_hops), budget
    )


# --- Sketch-based (over sampled live-edge graphs) --------------------------
def static_greedy(
    g: GraphInfo, budget: int, diffusion_model: str = "IC", snapshots: int = 20, **_
) -> list[int]:
    """StaticGreedy (Cheng et al. 2014): greedy over a fixed set of live-edge snapshots."""
    rng = np.random.default_rng(0)
    graphs = [P.sample_live_edge_graph(g, rng) for _ in range(snapshots)]
    seeds: list[int] = []
    for _ in range(budget):
        best_v, best_gain = -1, -1.0
        for v in range(g.num_nodes):
            if v in seeds:
                continue
            gain = float(
                np.mean(
                    [
                        P.reachable_count(lg, seeds + [v])
                        - P.reachable_count(lg, seeds)
                        for lg in graphs
                    ]
                )
            )
            if gain > best_gain:
                best_gain, best_v = gain, v
        seeds.append(best_v)
    return seeds


def skim(
    g: GraphInfo, budget: int, diffusion_model: str = "IC", snapshots: int = 32, **_
) -> list[int]:
    """SKIM (Cohen et al. 2014): rank by average single-seed reachability over sketches (simplified)."""
    rng = np.random.default_rng(0)
    graphs = [P.sample_live_edge_graph(g, rng) for _ in range(snapshots)]
    score = np.array(
        [
            float(np.mean([P.reachable_count(lg, [v]) for lg in graphs]))
            for v in range(g.num_nodes)
        ]
    )
    return _greedy_discount_select(g, score, budget)


# --- Community (additions) --------------------------------------------------
def cofim(g: GraphInfo, budget: int, diffusion_model: str = "IC", **_) -> list[int]:
    """CoFIM (Zhang et al. 2014): per-community budget + degree with cross-community bridge bonus."""
    comm = P.detect_communities(g)
    alloc = P.allocate_budget(comm, budget)
    members: dict[int, list[int]] = {}
    for v, c in comm.items():
        members.setdefault(c, []).append(v)
    deg = P.compute_degree(g)

    def bridge_score(v: int) -> float:
        nb = g.out_neighbors(v) + g.in_neighbors(v)
        return float(deg[v] + sum(1 for w in nb if comm[w] != comm[v]))

    seeds: list[int] = []
    for c, b in alloc.items():
        seeds.extend(sorted(members[c], key=bridge_score, reverse=True)[:b])
    return _pad_seeds(seeds, g, budget)


def community_ris(
    g: GraphInfo, budget: int, diffusion_model: str = "IC", theta: int = 2000, **_
) -> list[int]:
    """Community-RIS hybrid: RR-set coverage selection restricted to per-community budgets."""
    comm = P.detect_communities(g)
    alloc = P.allocate_budget(comm, budget)
    rr = P.batch_reverse_sample(g, theta=theta, seed=0)
    members: dict[int, list[int]] = {}
    for v, c in comm.items():
        members.setdefault(c, []).append(v)
    covers: dict[int, set[int]] = {v: set() for v in range(g.num_nodes)}
    for idx, s in enumerate(rr):
        for v in s:
            covers[v].add(idx)
    seeds: list[int] = []
    covered: set[int] = set()
    for c, b in alloc.items():
        for _ in range(b):
            best_v, best_gain = -1, -1
            for v in members[c]:
                if v in seeds:
                    continue
                gain = len(covers[v] - covered)
                if gain > best_gain:
                    best_gain, best_v = gain, v
            if best_v >= 0:
                seeds.append(best_v)
                covered |= covers[best_v]
    return _pad_seeds(seeds, g, budget)


# --- Metaheuristics (additions) --------------------------------------------
def simulated_annealing(
    g: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    mc_runs: int = 20,
    horizon: int = 20,
    iters: int = 40,
    **_,
) -> list[int]:
    """Simulated annealing over seed sets (degree init, swap moves, geometric cooling)."""
    rng = np.random.default_rng(0)
    seeds = P.get_top_degree_nodes(g, budget)
    cur = best = P.mc_simulate_spread(g, seeds, diffusion_model, mc_runs, horizon)
    best_seeds = list(seeds)
    temp = 1.0
    for _ in range(iters):
        cands = [v for v in range(g.num_nodes) if v not in seeds]
        if not cands:
            break
        trial = list(seeds)
        trial[int(rng.integers(len(seeds)))] = int(rng.choice(cands))
        sp = P.mc_simulate_spread(g, trial, diffusion_model, mc_runs, horizon)
        if sp > cur or rng.random() < np.exp((sp - cur) / max(temp, 1e-6)):
            seeds, cur = trial, sp
            if sp > best:
                best, best_seeds = sp, list(trial)
        temp *= 0.9
    return best_seeds


def genetic_algorithm(
    g: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    pop: int = 8,
    gens: int = 6,
    mc_runs: int = 15,
    horizon: int = 20,
    **_,
) -> list[int]:
    """Genetic algorithm over seed sets (degree-biased init, crossover + mutation, elitism)."""
    rng = np.random.default_rng(0)
    pool = P.get_top_degree_nodes(g, min(g.num_nodes, max(budget * 4, 10)))

    def fill(child: list[int]) -> list[int]:
        child = list(dict.fromkeys(child))
        while len(child) < budget:
            v = int(rng.choice(pool))
            if v not in child:
                child.append(v)
        return child[:budget]

    def fitness(ind: list[int]) -> float:
        return P.mc_simulate_spread(g, list(ind), diffusion_model, mc_runs, horizon)

    population = [
        fill(list(rng.choice(pool, size=budget, replace=False))) for _ in range(pop)
    ]
    scored = sorted(
        ((fitness(i), i) for i in population), key=lambda x: x[0], reverse=True
    )
    for _ in range(gens):
        survivors = [i for _, i in scored[: max(2, pop // 2)]]
        children = []
        while len(children) < pop - len(survivors):
            a = survivors[int(rng.integers(len(survivors)))]
            b = survivors[int(rng.integers(len(survivors)))]
            cut = budget // 2
            child = fill(a[:cut] + b[cut:])
            if rng.random() < 0.3:
                child[int(rng.integers(budget))] = int(rng.choice(pool))
                child = fill(child)
            children.append(child)
        population = survivors + children
        scored = sorted(
            ((fitness(i), i) for i in population), key=lambda x: x[0], reverse=True
        )
    return list(scored[0][1])


# --- Hybrids (additions) ----------------------------------------------------
def degree_ris_refine(
    g: GraphInfo, budget: int, diffusion_model: str = "IC", theta: int = 2000, **_
) -> list[int]:
    """Degree init refined by RR-set coverage swaps (Degree + RIS)."""
    seeds = P.get_top_degree_nodes(g, budget)
    rr = P.batch_reverse_sample(g, theta=theta, seed=0)
    covers: dict[int, set[int]] = {v: set() for v in range(g.num_nodes)}
    for idx, s in enumerate(rr):
        for v in s:
            covers[v].add(idx)
    covered: set[int] = set()
    for v in seeds:
        covered |= covers[v]
    for i, s in enumerate(seeds):
        rest = covered - covers[s]
        for v in range(g.num_nodes):
            if v in seeds:
                continue
            if len(covers[v] - rest) > len(covers[s] - rest):
                covered = rest | covers[v]
                seeds[i] = v
                break
    return seeds


def celf_local_search(
    g: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    mc_runs: int = 30,
    horizon: int = 20,
    **_,
) -> list[int]:
    """CELF seed set refined by 1-swap local search (CELF + LocalSearch)."""
    seeds = celf(g, budget, diffusion_model, mc_runs, horizon)
    best = P.mc_simulate_spread(g, seeds, diffusion_model, mc_runs, horizon)
    for i in range(len(seeds)):
        for v in range(g.num_nodes):
            if v in seeds:
                continue
            trial = list(seeds)
            trial[i] = v
            sp = P.mc_simulate_spread(g, trial, diffusion_model, mc_runs, horizon)
            if sp > best:
                best, seeds = sp, trial
                break
    return seeds


def community_celf(
    g: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    mc_runs: int = 20,
    horizon: int = 20,
    **_,
) -> list[int]:
    """Community-scoped marginal-gain greedy (no CELF lazy heap — simplified)."""
    comm = P.detect_communities(g)
    alloc = P.allocate_budget(comm, budget)
    members: dict[int, list[int]] = {}
    for v, c in comm.items():
        members.setdefault(c, []).append(v)
    seeds: list[int] = []
    for c, b in alloc.items():
        local: list[int] = []
        for _ in range(b):
            best_v, best_gain = -1, float("-inf")
            for v in members[c]:
                if v in local:
                    continue
                gain = P.compute_marginal_gain(
                    g, seeds + local, v, diffusion_model, mc_runs, horizon
                )
                if gain > best_gain:
                    best_gain, best_v = gain, v
            if best_v >= 0:
                local.append(best_v)
        seeds.extend(local)
    return _pad_seeds(seeds, g, budget)


# Registry the library API + README enumerate.
algorithms = {
    # degree
    "high_degree": high_degree,
    "weighted_degree": weighted_degree,
    "degree_discount": degree_discount,
    # centrality
    "pagerank_seeds": pagerank_seeds,
    "eigenvector_seeds": eigenvector_seeds,
    "closeness_seeds": closeness_seeds,
    # greedy
    "vanilla_greedy": vanilla_greedy,
    "celf": celf,
    "celf_pp": celf_pp,
    "adaptive_greedy": adaptive_greedy,
    # RIS
    "ris_basic": ris_basic,
    "tim": tim,
    "imm": imm,
    "ssa": ssa,
    "filtered_ris": filtered_ris,
    # path
    "sp1m": sp1m,
    "mia_pmia": mia_pmia,
    "ldag": ldag,
    # sketch
    "static_greedy": static_greedy,
    "skim": skim,
    # community
    "community_im": community_im,
    "cofim": cofim,
    "community_ris": community_ris,
    # metaheuristic
    "simulated_annealing": simulated_annealing,
    "hill_climbing": hill_climbing,
    "genetic_algorithm": genetic_algorithm,
    # hybrid
    "pagerank_greedy": pagerank_greedy,
    "degree_ris_refine": degree_ris_refine,
    "celf_local_search": celf_local_search,
    "community_celf": community_celf,
}
