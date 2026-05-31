import pytest

from stressnet.utils.plotting import _normalize_force_to_plot, plot_with_force


class _Vertex:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


class _Edge:
    def __init__(self, edge_id: int, v1: _Vertex, v2: _Vertex, tension: float, gt: float) -> None:
        self.id = edge_id
        self.v1 = v1
        self.v2 = v2
        self.tension = tension
        self.gt = gt


class _BigEdge:
    def __init__(self, edge_ids: list[int], external: bool = False) -> None:
        self.edges = edge_ids
        self.external = external


class _Frame:
    def __init__(self) -> None:
        vertices = [
            _Vertex(0.0, 0.0),
            _Vertex(1.0, 0.0),
            _Vertex(1.0, 1.0),
            _Vertex(0.0, 1.0),
        ]
        self.edges = {
            0: _Edge(0, vertices[0], vertices[1], tension=0.4, gt=0.5),
            1: _Edge(1, vertices[1], vertices[2], tension=0.8, gt=0.9),
            2: _Edge(2, vertices[2], vertices[3], tension=1.2, gt=1.1),
            3: _Edge(3, vertices[3], vertices[0], tension=1.0, gt=1.0),
        }
        self.big_edges = {
            0: _BigEdge([0, 1]),
            1: _BigEdge([2]),
            2: _BigEdge([3], external=True),
        }


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


def test_plot_with_force_runs_and_saves_continuous_colormap(tmp_path) -> None:
    output = tmp_path / 'stress_plot'

    plot_with_force(
        _Frame(),
        filename=str(output),
        force_to_plot='stress',
        cbar=True,
        title='StressNET prediction',
    )

    assert output.with_suffix('.png').is_file()
