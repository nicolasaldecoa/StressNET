"""Matplotlib helpers for visualizing ForSys frames and inferred forces."""

__all__ = ['plot_with_force']

import os
from typing import TYPE_CHECKING, Literal

import cmocean  # noqa: F401  # registers "cmo.*" colormaps with matplotlib
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from forsys.frames import Frame

CBAR_STEP = 0.2
DEFAULT_DISCRETE_CMAP = 'tab20'
DEFAULT_CONTINUOUS_CMAP = 'cmo.matter'


def _normalize_force_to_plot(force_to_plot: Literal['stress', 'tension', 'gt', 'ground-truth'] | None
                             ) -> Literal['stress', 'tension', 'gt', 'ground-truth']:
    if force_to_plot is None:
        return None
    if force_to_plot in ('stress', 'tension'):
        return 'tension'
    if force_to_plot in ('gt', 'ground-truth'):
        return 'gt'
    raise ValueError(f'"{force_to_plot}" not supported, must provide "stress", "tension", "gt", or "ground-truth".')


def plot_with_force(frame: 'Frame',
                    filename: str | None = None,
                    force_to_plot: Literal['stress', 'tension', 'gt', 'ground-truth'] | None = None,
                    mirror_y: bool = False,
                    figsize: tuple[float, float] = (10, 10),
                    **kwargs
                    ) -> None:
    """Plot the tissue graph, optionally coloring edges by tension or ground truth.

    Parameters
    ----------
    frame
        ForSys frame with vertices/edges populated.
    filename
        If set, save the figure to this path (``.png`` appended when extension missing).
    force_to_plot
        ``None`` draws discrete colors per internal big edge; ``'stress'`` / ``'tension'``
        maps continuous edge ``tension``; ``'gt'`` / ``'ground-truth'`` maps ``gt``.
    mirror_y
        If ``True``, invert the y-axis (image-style coordinates).
    figsize
        Matplotlib figure size in inches.
    **kwargs
        Common keys: ``cmap``, ``cbar``, ``cbar_step``, ``cbar_params``, ``title``,
        ``plot_kwargs`` (merged into ``plt.plot`` for non-external edges).

    Returns
    -------
    None
        Displays interactively when ``filename`` is ``None``; otherwise writes file and closes.
    """
    force_to_plot = _normalize_force_to_plot(force_to_plot)
    plt.close()

    _, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')

    if filename:
        new_dir = os.path.dirname(filename)
        if new_dir:
            os.makedirs(new_dir, exist_ok=True)

    cbar_step = kwargs.get('cbar_step', CBAR_STEP)

    external_edge_ids = {eid for be in frame.big_edges.values() for eid in be.edges if be.external}
    if force_to_plot is None:
        # discrete color mapping, one for each big edge
        colormap = plt.get_cmap(kwargs.get('cmap', DEFAULT_DISCRETE_CMAP))
        big_edges_to_color = [be for be in frame.big_edges.values() if not be.external]
        edge_id_to_color_index = {eid: i for i, be in enumerate(big_edges_to_color) for eid in be.edges}
        clean_vmax = cbar_norm = clean_vmin = None

        def _get_edge_color(e):
            return colormap(edge_id_to_color_index[e.id] % colormap.N)

    else:
        colormap = plt.get_cmap(kwargs.get('cmap', DEFAULT_CONTINUOUS_CMAP))
        all_forces = [getattr(edge, force_to_plot) for eid, edge in frame.edges.items() if eid not in external_edge_ids]
        # Calculate the 'clean' floor (e.g., 0.3 becomes 0.2 if CBAR_STEP=0.2)
        clean_vmin = np.floor(np.min(all_forces) / cbar_step) * cbar_step
        # Calculate the 'clean' ceiling (e.g., 1.7 becomes 1.8 if CBAR_STEP=0.2)
        clean_vmax = np.ceil(np.max(all_forces) / cbar_step) * cbar_step
        cbar_norm = matplotlib.colors.Normalize(vmin=clean_vmin, vmax=clean_vmax)

        def _get_edge_color(e):
            val = getattr(e, force_to_plot)
            return colormap(cbar_norm(val))

    for eid, edge in frame.edges.items():
        if eid in external_edge_ids:
            plot_kwargs = {'color': 'black', 'linewidth': 0.5, 'alpha': 0.6}
        else:
            plot_kwargs = {
                'color': _get_edge_color(edge),
                'linewidth': 2,
                **kwargs.get('plot_kwargs', {})
            }

        plt.plot((edge.v1.x, edge.v2.x),
                 (edge.v1.y, edge.v2.y),
                 **plot_kwargs)

    plt.axis('off')
    if mirror_y:
        plt.gca().invert_yaxis()

    if kwargs.get('cbar') and cbar_norm is not None:
        sm = matplotlib.cm.ScalarMappable(cmap=colormap, norm=cbar_norm)
        sm.set_array([])
        default_cbar_params = {'pad': 0.04, 'shrink': 0.7, 'format': '%.1f'}
        cbar_params = {**default_cbar_params, **kwargs.get('cbar_params', {})}
        cbar = plt.colorbar(sm, ax=ax, **cbar_params)
        ticks = np.arange(clean_vmin, clean_vmax + (cbar_step / 2), cbar_step)
        cbar.set_ticks(ticks)
        if kwargs.get('cbar_tick_params'):
            cbar.ax.tick_params(**kwargs['cbar_tick_params'])

    if kwargs.get('title'):
        plt.title(kwargs['title'])

    plt.tight_layout()
    if filename:
        filename = filename if filename.split('.')[-1] in ('png', 'pdf', 'svg') else (filename + '.png')
        plt.savefig(filename, dpi=500, transparent=True)
        plt.close()
    else:
        plt.plot()
