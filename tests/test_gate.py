import json

import pytest
from omegaconf import OmegaConf

from wm811k.gate import check_gate, passes_gate


def test_gate_passes_above_threshold():
    assert passes_gate({"test_macro_f1": 0.5}, min_macro_f1=0.4)


def test_gate_passes_exactly_at_threshold():
    assert passes_gate({"test_macro_f1": 0.4}, min_macro_f1=0.4)


def test_gate_fails_below_threshold():
    assert not passes_gate({"test_macro_f1": 0.3}, min_macro_f1=0.4)


def test_gate_fails_when_metric_missing():
    assert not passes_gate({"val_macro_f1": 0.9}, min_macro_f1=0.4)


@pytest.mark.parametrize(("f1", "expected"), [(0.6, True), (0.1, False)])
def test_check_gate_reads_the_committed_metadata_file(tmp_path, f1, expected):
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"test_macro_f1": f1}))
    cfg = OmegaConf.create({"gate": {"min_macro_f1": 0.4, "metadata_file": str(metadata)}})

    assert check_gate(cfg) is expected


def test_check_gate_fails_when_the_metadata_file_is_missing(tmp_path):
    cfg = OmegaConf.create(
        {"gate": {"min_macro_f1": 0.4, "metadata_file": str(tmp_path / "nope.json")}}
    )

    assert check_gate(cfg) is False
