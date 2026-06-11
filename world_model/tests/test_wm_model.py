import sys
sys.path.insert(0, "world_model")
import numpy as np, torch
from wm_data import build_graph_input
from wm_model import WorldModel, BACKBONES


def test_all_backbones_forward():
    ei = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int64)
    w = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    gi = build_graph_input(ei, w, 3, "IC", torch.device("cpu"))
    X = torch.rand(3, 6)
    for name in BACKBONES:
        out = WorldModel(name, in_channels=6, hidden_dim=16, n_layers=2)(X, gi)
        assert out.shape == (3, 2), name
