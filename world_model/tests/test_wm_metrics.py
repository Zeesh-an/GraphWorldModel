import sys; sys.path.insert(0, "world_model")
import numpy as np
from wm_metrics import binary_f1, new_infection_f1, delta_f1


def test_binary_f1_perfect_and_empty():
    assert binary_f1(np.array([1, 0, 1]), np.array([1, 0, 1])) == 1.0
    # no positives in either -> define F1 = 1.0 (nothing to find, nothing predicted)
    assert binary_f1(np.array([0, 0]), np.array([0, 0])) == 1.0


def test_new_infection_f1_restricts_to_susceptible():
    infected_t = np.array([1, 0, 0, 0])     # node 0 already infected
    y_inf      = np.array([1, 1, 0, 0])     # node 1 newly infected
    pred_inf   = np.array([1, 1, 1, 0])     # predicts 1 (correct) and 2 (false positive)
    # newly-infected truth = {1}; pred-new = {1,2}; precision .5, recall 1 -> F1 = 2/3
    assert abs(new_infection_f1(pred_inf, y_inf, infected_t) - (2/3)) < 1e-6
