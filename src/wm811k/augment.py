import torch

DIE_DROPOUT = 0.05


def wafer_augment(x: torch.Tensor) -> torch.Tensor:
    """Random 90-degree rotation, flip and defective-die dropout on a one-hot (B, 3, H, W) batch.

    Wafers have no canonical orientation, so rotations and flips keep the class. The output stays
    one-hot: dropped defective dies become good dies.
    """
    out = x.clone()
    n, _, h, w = out.shape
    quarter_turns = torch.randint(0, 4, (n,), device=out.device)
    for k in (1, 2, 3):
        chosen = quarter_turns == k
        out[chosen] = torch.rot90(out[chosen], k, dims=(2, 3))
    flipped = torch.rand(n, device=out.device) < 0.5
    out[flipped] = out[flipped].flip(3)

    drop_draw = torch.rand(n, 1, h, w, device=out.device) < DIE_DROPOUT
    dropped = (drop_draw & (out[:, 2:3] > 0)).float()
    out[:, 1:2] += dropped
    out[:, 2:3] -= dropped
    return out


def weak_augment(x: torch.Tensor) -> torch.Tensor:
    """Random vertical flip of a one-hot (B, 3, H, W) batch; the mild view FixMatch labels from."""
    out = x.clone()
    flipped = torch.rand(out.shape[0], device=out.device) < 0.5
    out[flipped] = out[flipped].flip(2)
    return out
