import math
import torch

# Standard sinusoidal-encoding wavelength base (Vaswani et al. 2017)
sinusoid_base = 10000.0


def degree_encoding(
    adjacency: torch.Tensor, d_model: int, device: torch.device
) -> torch.Tensor:
    """
    Simple, fixed degree-based positional encodings.
    Encodes log(1 + degree) projected to d_model dims via a fixed sinusoidal scheme.

    adjacency: sparse COO (N, N) or dense (N, N)

    Returns: (N, d_model) float tensor
    """
    # Compute the degree of each node in the graph adjacency matrix
    if adjacency.is_sparse:
        degrees = torch.sparse.sum(adjacency, dim=1).to_dense()  # (N,)
    else:
        degrees = adjacency.sum(dim=1)

    degrees = torch.log1p(degrees).unsqueeze(dim=1)  # (N, 1)

    # Sinusoidal projection across d_model dimensions for positional encodings
    frequencies = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float, device=device)
        * -(math.log(sinusoid_base) / d_model)
    )  # (d_model / 2,)

    encoding = torch.zeros(degrees.shape[0], d_model, device=device)
    encoding[:, 0::2] = torch.sin(degrees * frequencies)
    encoding[:, 1::2] = torch.cos(degrees * frequencies[: d_model // 2])

    return encoding  # (N, d_model)
