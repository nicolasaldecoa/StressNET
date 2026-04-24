import pytest

from stressnet.utils.plotting import _normalize_force_to_plot


@pytest.mark.parametrize('alias', ['stress', 'tension'])
def test_normalize_force_to_plot_tension_aliases(alias) -> None:
    assert _normalize_force_to_plot(alias) == 'tension'


@pytest.mark.parametrize('alias', ['gt', 'ground-truth'])
def test_normalize_force_to_plot_ground_truth_aliases(alias) -> None:
    assert _normalize_force_to_plot(alias) == 'gt'


def test_normalize_force_to_plot_accepts_none() -> None:
    assert _normalize_force_to_plot(None) is None


def test_normalize_force_to_plot_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match='not supported'):
        _normalize_force_to_plot('pressure')
