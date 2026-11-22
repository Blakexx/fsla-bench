"""The FSLA arms registry (`fsla_bench.models.fsla`): every build of a cell is the same layer, built only through
fla, and its state is read back off that layer. CPU only."""
from __future__ import annotations

import pytest
import torch

from fsla_bench.models import fsla as cells


def _params(module):
    return dict(module.named_parameters())


@pytest.mark.parametrize("wiring", sorted(cells.WIRINGS))
def test_the_mixer_and_the_bare_layer_are_the_same_layer(wiring):
    """zoology's mixer config and the bare fla layer build bit-identical parameters from one cell."""
    from zoology.mixers.fsla import FSLAMixer

    c = cells.cell(wiring, 64)
    torch.manual_seed(0)
    layer = cells.layer(c, hidden_size=128, num_heads=2, d_v=64)
    torch.manual_seed(0)
    config = cells.mixer_config(c, n_heads=2, d_v=64)
    mixer = FSLAMixer(d_model=128, layer_idx=0, **config["kwargs"])
    a, b = _params(layer), _params(mixer.layer)
    assert sorted(a) == sorted(b) and all(torch.equal(a[k], b[k]) for k in a)
    assert config["name"] == "zoology.mixers.fsla.FSLAMixer"


def test_a_cell_is_fslas_own_layer_with_the_wirings_levels():
    import fsla

    c = cells.cell("fsla-hybrid", 256)
    layer = cells.layer(c, hidden_size=128, num_heads=2, d_v=64)
    assert type(layer.layer) is fsla.FSLA
    assert layer.topology.widths == (16, 16)
    assert [type(level) for level in layer.routes.levels] == [fsla.UnionRouting, fsla.IndependentRouting]


def test_state_is_read_off_the_built_layer():
    for n in (16, 64, 256):
        layer = cells.layer(cells.cell("fsla-arm2-union", n), hidden_size=128, num_heads=4, d_v=64)
        assert cells.state_floats(layer) == (4 * n * 64, 4 * n)


def test_a_cells_decay_is_fslas_source_built_with_its_widths():
    import fsla

    c = cells.cell("fsla-arm1-densread-sparsewrite", 128, widths=(8, 16), decay={"type": "LearnedDecay"})
    built = cells.layer(c, hidden_size=128, num_heads=2, d_v=64)
    assert isinstance(built.layer.decay, fsla.LearnedDecay) and built.layer.decay.widths == (8, 16)
    assert cells.cell("fsla-arm1-densread-sparsewrite", 256).decay is None


def test_an_unknown_wiring_or_a_mismatched_width_is_refused():
    with pytest.raises(ValueError, match="wiring"):
        cells.Cell("fsla-arm9", (16,))
    with pytest.raises(ValueError, match="level"):
        cells.Cell("fsla-d1-dense", (8, 8))
