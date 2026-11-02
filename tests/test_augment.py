import torch

from wm811k.augment import wafer_augment


def _one_hot_batch(n=32, size=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, 3, (n, size, size), generator=g)
    return torch.nn.functional.one_hot(idx, 3).permute(0, 3, 1, 2).float()


def test_augment_keeps_one_hot_validity():
    out = wafer_augment(_one_hot_batch())

    assert out.shape == (32, 3, 16, 16)
    assert torch.equal(out.sum(dim=1), torch.ones(32, 16, 16))
    assert set(out.unique().tolist()) <= {0.0, 1.0}


def test_augment_changes_input_for_some_draws():
    x = _one_hot_batch()

    assert not torch.equal(wafer_augment(x), x)


def test_augment_is_deterministic_under_torch_seed():
    x = _one_hot_batch()
    torch.manual_seed(5)
    a = wafer_augment(x)
    torch.manual_seed(5)
    b = wafer_augment(x)

    assert torch.equal(a, b)


def test_augment_does_not_modify_its_input():
    x = _one_hot_batch()
    before = x.clone()

    wafer_augment(x)

    assert torch.equal(x, before)
