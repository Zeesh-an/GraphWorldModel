import math
import torch


# Fixed, degree-based Positional Encoding
def degree_encoding(
    adj: torch.Tensor, d_model: int, device: torch.device
) -> torch.Tensor:
    """
    Simple degree-based positional encoding.
    Encodes log(1 + degree) projected to d_model dims via a fixed sinusoidal scheme.

    adj: sparse COO (N, N) or dense (N, N)

    Returns: (N, d_model) float tensor
    """

    # Compute the degree of each node in the graph adjacency matrix
    if adj.is_sparse:
        deg = torch.sparse.sum(adj, dim=1).to_dense()  # (N,)
    else:
        deg = adj.sum(dim=1)

    deg = torch.log1p(deg).unsqueeze(dim=1)  # (N, 1)

    # Sinusoidal projection across d_model dimensions for positional encodings
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float, device=device)
        * -(math.log(10000.0) / d_model)
    )  # (d_model / 2,)

    pe = torch.zeros(deg.shape[0], d_model, device=device)
    pe[:, 0::2] = torch.sin(deg * div)
    pe[:, 1::2] = torch.cos(deg * div[: d_model // 2])

    return pe  # (N, d_model)
