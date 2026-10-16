"""The two classifiers. Both take a batch of point clouds shaped (B, N, 3) and
return unnormalised class scores (logits) shaped (B, 10).

A "shared" layer applies the same weights to every point independently. It is
written as ``nn.Conv1d(in, out, kernel_size=1)`` over the (B, channels, N)
layout, which is exactly a per-point ``nn.Linear`` and lets BatchNorm1d
normalise each channel over the batch and all points. Global max pooling over
the N points makes the result independent of point order.
"""

from __future__ import annotations

from torch import Tensor, nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.2),
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "tanh": nn.Tanh,
}
MODEL_NAMES = ("shallow", "mini_pointnet")


def make_activation(name: str) -> nn.Module:
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation {name!r}; choose one of {sorted(ACTIVATIONS)}")
    return ACTIVATIONS[name]()


class ShallowPointNet(nn.Module):
    """Benchmark: one shared 3→h layer, activation, global max pool, linear classifier."""

    def __init__(self, num_classes: int = 10, hidden_dim: int = 64, activation: str = "relu"):
        super().__init__()
        self.point_layer = nn.Conv1d(3, hidden_dim, kernel_size=1)
        self.activation = make_activation(activation)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, points: Tensor) -> Tensor:
        x = points.transpose(1, 2)                  # (B, 3, N)
        x = self.activation(self.point_layer(x))    # (B, h, N): one hidden layer, per point
        x = x.max(dim=2).values                     # (B, h): global max pool over points
        return self.classifier(x)                   # (B, classes): logits


class MiniPointNet(nn.Module):
    """Shared MLP 3→h→2h→4h with BatchNorm, global max pool, head 4h→2h→classes.

    With the default ``hidden_dim=64`` this is 3→64→128→256, max pool, 256→128→10.
    """

    def __init__(self, num_classes: int = 10, hidden_dim: int = 64, activation: str = "relu",
                 dropout: float = 0.3):
        super().__init__()
        widths = [3, hidden_dim, 2 * hidden_dim, 4 * hidden_dim]
        layers: list[nn.Module] = []
        for c_in, c_out in zip(widths[:-1], widths[1:]):
            layers += [nn.Conv1d(c_in, c_out, kernel_size=1, bias=False), nn.BatchNorm1d(c_out), make_activation(activation)]
        self.point_mlp = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(4 * hidden_dim, 2 * hidden_dim, bias=False),
            nn.BatchNorm1d(2 * hidden_dim),
            make_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, num_classes),
        )

    def forward(self, points: Tensor) -> Tensor:
        x = self.point_mlp(points.transpose(1, 2))  # (B, 4h, N): three hidden layers, per point
        x = x.max(dim=2).values                     # (B, 4h): global feature
        return self.head(x)                         # (B, classes): logits


def build_model(name: str, num_classes: int = 10, hidden_dim: int = 64, activation: str = "relu",
                dropout: float = 0.3) -> nn.Module:
    if name == "shallow":
        return ShallowPointNet(num_classes, hidden_dim, activation)
    if name == "mini_pointnet":
        return MiniPointNet(num_classes, hidden_dim, activation, dropout)
    raise ValueError(f"Unknown model {name!r}; choose one of {MODEL_NAMES}")


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
