import numpy as np

from coding_agent.tools.primitives import ris_select


def _reference(rr_sets: list[set[int]], budget: int, num_nodes: int) -> list[int]:
    # The per-node set-difference scan this replaced, kept as the oracle
    covers = {node: set() for node in range(num_nodes)}
    for rr_index, rr_set in enumerate(rr_sets):
        for node in rr_set:
            covers[node].add(rr_index)
    chosen, covered = [], set()
    for _ in range(budget):
        best_node, best_gain = -1, -1
        for node in range(num_nodes):
            if node in chosen:
                continue
            gain = len(covers[node] - covered)
            if gain > best_gain:
                best_gain, best_node = gain, node
        if best_node < 0:
            break
        chosen.append(best_node)
        covered |= covers[best_node]
    return chosen


def test_counting_selection_matches_the_scan_including_ties_and_overrun() -> None:
    rng = np.random.default_rng(3)
    for trial in range(30):
        num_nodes = int(rng.integers(1, 30))
        rr_sets = [set(rng.integers(0, num_nodes, size=int(rng.integers(0, 6))).tolist()) for _ in range(int(rng.integers(0, 40)))]
        for budget in (0, 1, 3, num_nodes, num_nodes + 4):
            assert ris_select(rr_sets, budget, num_nodes) == _reference(rr_sets, budget, num_nodes), (trial, budget)
