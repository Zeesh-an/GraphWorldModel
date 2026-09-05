"""
Self-describing world-model checkpoints.

Until now `train_wm.py` wrote a bare `state_dict`. Everything needed to rebuild
the module it belonged to — backbone, head, widths, `remove_semantics`,
`action_encoding`, `hide_edge_weights` — lived only in the sibling results JSON
and in the CLI flags of the run. Three consequences, all of them real:

  * a checkpoint could not be loaded from the checkpoint. `WorldModelScorer.load`
    was impossible, and `eval_planning` / `world_model_env` each carried their own
    copy of the reconstruct-from-config logic, free to drift apart.

  * a `state_dict` from a `sage` run loaded into a `gcn` model raises a shape
    error, which is survivable — but a `spent` checkpoint loaded under `blocked`
    does NOT raise. The tensors are identically shaped. It silently evaluates a
    head whose T_exo contradicts the data it was fit against.

  * moving a checkpoint without its results JSON lost the architecture entirely.

This module makes the file self-contained. `ModelSpec` is exactly the set of
fields `WorldModel.__init__` consumes, so "can this checkpoint be rebuilt" is a
property of the file rather than of its neighbours.

Backward compatibility is explicit, not silent. A v1 (bare `state_dict`) file
still loads — but only when the caller supplies the config, because the
information genuinely is not in the file. `load_checkpoint` says so in the error
rather than guessing defaults that would quietly reproduce the wrong model.
"""

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import torch
import torch.nn as nn

from data.wm_competitive import auto_dominance
from data.wm_simulator import spent, valid_remove_semantics
from world_model.wm_data import basic_encoding, channels_for, num_input_channels
from world_model.wm_model import WorldModel, backbones

#: Bumped when the blob layout changes. v1 is the historical bare state_dict,
#: which has no marker of its own — it is recognised by the ABSENCE of this key.
checkpoint_format = "wm-ckpt-v2"

legacy_format = "wm-ckpt-v1-bare-state-dict"

valid_heads = ("linear", "structured", "structured_residual", "structured_oracle")


@dataclass(frozen=True)
class ModelSpec:
    """
    Everything `WorldModel.__init__` needs, and nothing else.

    Deliberately NOT the whole `TrainConfig`: learning rate, epochs and patience
    describe how the weights were produced, not what module they belong in. They
    travel in the checkpoint's `train_meta` block, where they are provenance
    rather than a load-bearing part of reconstruction.

    `hide_edge_weights` and `action_encoding` ARE here even though they are data
    plumbing rather than module arguments: `action_encoding` decides `in_channels`
    (a `typed` checkpoint has a 9-wide input projection and will not load into a
    6-wide one), and `hide_edge_weights` decides what the encoder was fit to
    expect. Loading a w-hidden checkpoint and then feeding it true edge weights
    is silent, and it is exactly the confusion the hide-edge-weights experiment
    exists to resolve.
    """

    backbone: str
    head: str
    diffusion_model: str
    hidden_dim: int = 64
    n_layers: int = 3
    dropout: float = 0.1
    remove_semantics: str = spent
    action_encoding: str = basic_encoding
    hide_edge_weights: bool = False
    # Backbone-specific; ignored by backbones that do not read them, which is why
    # they can carry defaults rather than being optional.
    n_heads: int = 4
    ffn_dim: int = 128
    gcnii_alpha: float = 0.1
    gcnii_lamda: float = 0.5
    # Layout discriminators (influence blocking / epidemic control). They change
    # in_channels and which head class the name selects, so a checkpoint from one
    # layout can never silently load into another.
    competitive: bool = False
    tie_break: str = auto_dominance
    positive_prob: float | None = None
    epidemic: bool = False
    # The rates the compartmental head was fit under. Part of the spec rather than
    # train_meta because the oracle head pins its transition matrix to them: a
    # checkpoint rebuilt with different rates is a different model.
    beta_scale: float = 1.0
    gamma: float | None = None
    alpha: float | None = None

    def __post_init__(self) -> None:
        if self.backbone not in backbones:
            raise ValueError(
                f"unknown backbone {self.backbone!r}; choose from {list(backbones)}"
            )

        if self.head not in valid_heads:
            raise ValueError(
                f"unknown head {self.head!r}; choose from {list(valid_heads)}"
            )

        if self.remove_semantics not in valid_remove_semantics:
            raise ValueError(
                f"unknown remove_semantics {self.remove_semantics!r}; "
                f"choose from {list(valid_remove_semantics)}"
            )

    @property
    def in_channels(self) -> int:
        if self.competitive or self.epidemic:
            return channels_for(self.competitive, self.epidemic)[0]

        return num_input_channels(self.action_encoding)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, blob: dict) -> "ModelSpec":
        """
        Build from a dict, ignoring unknown keys and filling absent ones.

        Tolerant on purpose: this is what reads a results JSON `config` block,
        which carries the whole TrainConfig, and what reads checkpoints written
        by a future version that added a field.
        """
        known = {field.name for field in fields(cls)}

        return cls(**{key: value for key, value in blob.items() if key in known})

    @classmethod
    def from_train_config(cls, config) -> "ModelSpec":
        """
        Build from a `train_wm.TrainConfig`.

        `TrainConfig.model` is the backbone; the name differs because the results
        JSON schema has called it `model` since before there was a head to
        distinguish it from, and `world_model_env.from_results_json` reads that
        key. Renaming it would break every existing results JSON.
        """
        return cls(
            backbone=config.model,
            head=config.head,
            diffusion_model=config.diffusion_model,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            dropout=config.dropout,
            remove_semantics=config.remove_semantics,
            action_encoding=config.action_encoding,
            hide_edge_weights=config.hide_edge_weights,
            n_heads=config.n_heads,
            ffn_dim=config.ffn_dim,
            gcnii_alpha=config.gcnii_alpha,
            gcnii_lamda=config.gcnii_lamda,
            competitive=bool(config.competitive),
            tie_break=config.tie_break,
            positive_prob=config.positive_prob,
            epidemic=bool(config.epidemic),
            beta_scale=config.beta_scale,
            gamma=config.gamma,
            alpha=config.alpha,
        )

    @classmethod
    def from_results_json(cls, path: str | Path) -> "ModelSpec":
        """
        Build from a `train_wm.py` results JSON — the v1 compatibility path.

        The two `.get` defaults reproduce the behaviour of runs from before those
        flags existed, matching what `eval_planning.load_trained_model` and
        `world_model_env.from_results_json` already assumed. They are stated here
        once instead of in each caller.
        """
        return cls.from_config(json.loads(Path(path).read_text())["config"])

    @classmethod
    def from_config(cls, config: dict) -> "ModelSpec":
        """A results-JSON `config` block (or any TrainConfig-shaped dict), with the
        pre-flag defaults filled in. Shared by the path and dict entry points so
        neither can drift."""
        blob = dict(config)
        blob.setdefault("backbone", config.get("model"))
        blob.setdefault("remove_semantics", spent)
        blob.setdefault("action_encoding", basic_encoding)
        blob.setdefault("hide_edge_weights", False)
        blob.setdefault("head", "linear")

        return cls.from_dict(blob)


def build_model(spec: ModelSpec, device: torch.device | str = "cpu") -> nn.Module:
    """
    The single place a `WorldModel` is reconstructed from a spec.

    `eval_planning`, `world_model_env` and `WorldModelScorer` all route through
    here, so there is one definition of "what this checkpoint's architecture is"
    rather than three that can drift.
    """
    return WorldModel(
        spec.backbone,
        in_channels=spec.in_channels,
        hidden_dim=spec.hidden_dim,
        n_layers=spec.n_layers,
        dropout=spec.dropout,
        head_type=spec.head,
        diffusion_model=spec.diffusion_model,
        remove_semantics=spec.remove_semantics,
        competitive=spec.competitive,
        tie_break=spec.tie_break,
        positive_prob=spec.positive_prob,
        epidemic=spec.epidemic,
        epi_beta=spec.beta_scale,
        epi_gamma=spec.gamma,
        epi_alpha=spec.alpha,
        n_heads=spec.n_heads,
        ffn_dim=spec.ffn_dim,
        alpha=spec.gcnii_alpha,
        lamda=spec.gcnii_lamda,
    ).to(device)


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    spec: ModelSpec,
    train_meta: dict | None = None,
) -> Path:
    """
    Write a self-describing checkpoint.

    `train_meta` is provenance — dataset dir, split mode, seed, epochs, the
    registry digest, the git commit. It is never read during reconstruction, so a
    missing or malformed entry can never change which model comes back.
    """
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    torch.save(
        {
            "format": checkpoint_format,
            "spec": spec.to_dict(),
            "state_dict": model.state_dict(),
            "train_meta": dict(train_meta or {}),
        },
        path,
    )

    return path


class LegacyCheckpointError(ValueError):
    """
    A v1 bare-state_dict checkpoint was loaded without the config it needs.

    Its own error type so a caller can offer the results JSON rather than having
    to string-match, and so the recovery instructions live in one place.
    """


def read_checkpoint(path: str | Path) -> dict:
    """
    Load the raw blob and normalise both formats into one shape.

    Returns `{"format", "spec"|None, "state_dict", "train_meta"}`. `spec` is None
    exactly when the file is v1, which is the one case a caller must handle.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    # weights_only=False because a v2 blob carries plain dicts alongside the
    # tensors. The file is produced by this repository's own training runs, which
    # is the trust boundary torch's flag is about.
    blob = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(blob, dict) and blob.get("format") == checkpoint_format:
        return {
            "format": checkpoint_format,
            "spec": ModelSpec.from_dict(blob["spec"]),
            "state_dict": blob["state_dict"],
            "train_meta": blob.get("train_meta", {}),
        }

    # v1: the whole file IS the state dict. A tensor-valued mapping is the
    # signature; anything else is a file this loader does not understand.
    if isinstance(blob, dict) and all(
        isinstance(value, torch.Tensor) for value in blob.values()
    ):
        return {
            "format": legacy_format,
            "spec": None,
            "state_dict": blob,
            "train_meta": {},
        }

    raise ValueError(
        f"{path} is neither a {checkpoint_format} checkpoint nor a bare "
        f"state_dict; got {type(blob).__name__}"
    )


def load_checkpoint(
    path: str | Path,
    config: str | Path | dict | ModelSpec | None = None,
    device: torch.device | str = "cpu",
    strict_spec: bool = True,
) -> tuple[nn.Module, ModelSpec, dict]:
    """
    Rebuild the model a checkpoint belongs to. Returns `(model, spec, train_meta)`.

    `config` is only consulted for v1 files, or when `strict_spec=False` asks for
    an explicit override. It accepts a results-JSON path, a config dict, or a
    `ModelSpec`.

    On a v2 file with a `config` that DISAGREES, the default (`strict_spec=True`)
    raises. Silently preferring one over the other is how a `spent` checkpoint
    ends up evaluated as `blocked`: both load, neither errors, and only the
    numbers are wrong.
    """
    blob = read_checkpoint(path)
    supplied = _coerce_spec(config)

    if blob["spec"] is None:
        if supplied is None:
            raise LegacyCheckpointError(
                f"{path} is a legacy bare-state_dict checkpoint ({legacy_format}): "
                f"its architecture is not in the file, so it cannot be loaded "
                f"alone. Pass the run's results JSON, e.g.\n"
                f"    WorldModelScorer.load({str(path)!r}, "
                f"config='<run>/world_model/<backbone>_<dynamics>.json')\n"
                f"or retrain to emit a {checkpoint_format} checkpoint."
            )

        spec = supplied
    else:
        spec = blob["spec"]

        if supplied is not None and supplied != spec and strict_spec:
            raise ValueError(
                f"{path} describes itself as {spec}, but the supplied config says "
                f"{supplied}. Refusing to guess which is right: a mismatched "
                f"remove_semantics or hide_edge_weights loads without error and "
                f"changes every number. Pass strict_spec=False to override "
                f"deliberately."
            )

        if supplied is not None and not strict_spec:
            spec = supplied

    model = build_model(spec, device=device)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    return model, spec, blob["train_meta"]


def _coerce_spec(config) -> ModelSpec | None:
    if config is None:
        return None

    if isinstance(config, ModelSpec):
        return config

    if isinstance(config, dict):
        return ModelSpec.from_config(config)

    return ModelSpec.from_results_json(config)


def describe(path: str | Path) -> str:
    """One-line summary for a log or a CLI. Never raises on a legacy file."""
    blob = read_checkpoint(path)

    if blob["spec"] is None:
        return f"{Path(path).name}: {legacy_format} (architecture not in file)"

    spec = blob["spec"]
    meta = blob["train_meta"]

    return (
        f"{Path(path).name}: {spec.backbone}/{spec.head} {spec.diffusion_model} "
        f"hidden={spec.hidden_dim} layers={spec.n_layers} "
        f"remove={spec.remove_semantics} encoding={spec.action_encoding} "
        f"hide_w={spec.hide_edge_weights} "
        f"[split_mode={meta.get('split_mode', 'unknown')} "
        f"seed={meta.get('seed', 'unknown')}]"
    )
