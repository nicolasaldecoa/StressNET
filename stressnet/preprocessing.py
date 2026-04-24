"""Build Spektral-style graph tensors from ForSys / Surface Evolver workflows."""

__all__ = ['se_output_to_graph', 'skeleton_to_graph']

from .utils.data_utils import ConnectedNodes, resample_vertices
from .utils.plotting import plot_with_force
import forsys as fs
import numpy as np
from scipy.spatial.distance import euclidean as euclidean_dist
from scipy.sparse import dok_matrix
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from time import perf_counter
from logging import getLogger
import random
from typing import Any, Dict, List

StrPath = str | Path

log = getLogger(__name__)

NODE_FEATURES = ['arc_length', 'chord_length', 'cell1_area', 'cell2_area', 'cell1_per', 'cell2_per']


def _get_neighboring_vertices(vertex: fs.vertex.Vertex,
                              lattice: fs.surface_evolver.SurfaceEvolver
                              ) -> List[fs.vertex.Vertex]:
    neighbors = []
    # loop over edges connected to this vertex
    for eid in vertex.ownEdges:
        edge = lattice.edges[eid]
        # add the other vertex that's connected to each edge
        neighbor = edge.v1 if (edge.v1.id != vertex.id) else edge.v2
        neighbors.append(neighbor)
    return neighbors


def _jitter_vertices(lattice: fs.surface_evolver.SurfaceEvolver,
                     scale: float,
                     skip_big_edge_vertices: bool = True,
                     random_seed: int = None
                     ) -> None:
    rng = random.Random(random_seed)
    for v in lattice.vertices.values():
        if skip_big_edge_vertices and (len(v.ownEdges) != 2):
            continue  # only jitter vertices connected by 2 edges
        nvs = _get_neighboring_vertices(v, lattice)
        # calculate additive noise stdev as the scale factor multiplied
        # by the distance between the vertex and its closest neighbor
        stdev = scale * min(euclidean_dist([v.x, v.y], [nv.x, nv.y]) for nv in nvs)
        # add gaussian noise to each coordinate
        v.x += rng.normalvariate(0, stdev)
        v.y += rng.normalvariate(0, stdev)


def _forsys_frame_to_graph(frame: fs.frames.Frame,
                           include_targets: bool,
                           edge_n_vertices: int,
                           apply_savgol_filter: bool,
                           include_forsys_predictions: bool,
                           forsys_solve_method: str,
                           render_plots: bool,
                           plots_dir: StrPath,
                           return_timers: bool,
                           tag_cell_interfaces: list | None,
                           raise_if_gt_is_zero: bool,
                           load_time: float,
                           plots_prefix: str,
                           return_frame: bool,
                           verbose: bool
                           ) -> Dict[str, Any]:
    st = perf_counter()

    # optionally apply filter to remove noise
    if apply_savgol_filter:
        frame.filter_edges(method='SG')

    n_big_edges = len(frame.internal_big_edges)

    # predict tensions with forsys using this frame only
    forsys_pred_time = None
    if include_forsys_predictions:
        if verbose:
            log.info('Predicting tensions with forsys...')
        ft = perf_counter()
        forsys_engine = fs.ForSys({0: frame})
        forsys_engine.build_force_matrix(when=0, angle_limit=np.inf)
        forsys_engine.solve_stress(when=0, allow_negatives=False, method=forsys_solve_method)
        forsys_pred_time = perf_counter() - ft

    if verbose:
        log.info('Extracting features...')
    # big-edge vertex tracker to help us build the adjacency matrix later
    adj_tracker = defaultdict(set)
    # container for the big-edge points
    node_points = np.empty((n_big_edges, edge_n_vertices, 2), dtype=np.float32)
    # node features
    node_features = np.empty((n_big_edges, len(NODE_FEATURES)), dtype=np.float32)
    # ground-truths and forsys predictions
    ground_truth = np.empty(n_big_edges, dtype=np.float32) if include_targets else None
    forsys_preds = np.empty(n_big_edges, dtype=np.float32) if include_forsys_predictions else None
    # boolean array to tag specific big-edges
    tagged_nodes = np.empty(n_big_edges, dtype=np.bool) if tag_cell_interfaces else None
    to_tag = set([tuple(sorted(cell_pair)) for cell_pair in tag_cell_interfaces]) if tag_cell_interfaces else None

    bigi = 0
    for bigedge in frame.big_edges.values():
        if bigedge.external:
            continue

        if include_targets:
            assert (bigedge.gt > 0) or (not raise_if_gt_is_zero), f'Big-edge #{bigi} gt tension is {bigedge.gt:.4f}'
            ground_truth[bigi] = bigedge.gt

        if include_forsys_predictions:
            tension = bigedge.tension
            if tension < 0:
                log.warning(f'ForSys predicted tension for big-edge #{bigi} is negative: {tension:.4f}.')
            forsys_preds[bigi] = tension

        smedges = bigedge.edges  # list of small-edges ids
        n_smedges = len(smedges)

        # initialize the array of vertex coordinates
        n_points = n_smedges + 1
        points = np.empty((n_points, 2), dtype=np.float32)

        c1a = c2a = c1p = c2p = None
        is_tagged = False
        for smalli, edge_id in enumerate(smedges):
            # get edge
            edge = frame.edges[edge_id]

            # get the coordinates of the segment
            v1 = [edge.v1.x, edge.v1.y]
            v2 = [edge.v2.x, edge.v2.y]

            # store vertex #1 coordinates
            points[smalli] = v1

            # in the first one, store vertex_id and small edge coords
            if smalli == 0:
                assert len(edge.v1.ownEdges) > 2, f'Expecting tip of Big-edge #{bigi} to be connected to at least 3 ' \
                                                  f'edges, got {edge.v1.ownEdges}'
                adj_tracker[bigi].add(edge.v1.id)

                # get adjacent cell features from the second vertex of the first small-edge (not a triple junction)
                if n_smedges >= 2:
                    cells = edge.v2.ownCells
                else:  # unless it has only 1 small age and in that case we get cells that are common to both
                    cells = list(set(edge.v1.ownCells).intersection(edge.v2.ownCells))

                assert len(cells) <= 2, f'Expecting a maximum of 2 ownCells for vertex {edge.v2}, got {cells}.'
                c1 = frame.cells[cells[0]]
                c1a = abs(c1.get_area())
                c1p = c1.get_perimeter()
                if len(cells) > 1:
                    c2 = frame.cells[cells[1]]
                    c2a = abs(c2.get_area())
                    c2p = c2.get_perimeter()
                else:  # if we know of only one cell, replicate its data for the second one
                    c2a, c2p = c1a, c1p

                # check if we need to tag this big-edge
                if (to_tag is not None) and (tuple(sorted(cells)) in to_tag):
                    is_tagged = True

            # in the last one also store vertex #2 coordinates, vertex_id and small edge coords
            if smalli == (n_smedges - 1):
                assert len(edge.v2.ownEdges) > 2, f'Expecting tip of Big-edge #{bigi} to be connected to at least 3 ' \
                                                  f'edges, got {edge.v2.ownEdges}'
                points[-1] = v2  # store vertex #2 coordinates
                adj_tracker[bigi].add(edge.v2.id)

        if n_points != edge_n_vertices:
            # use spline interpolation to create intermediate points
            points = resample_vertices(points, target_length=edge_n_vertices, spline_order=2)
            operation = 'upsampled' if (edge_n_vertices > n_points) else 'downsampled'
            log.info(f'Big-edge #{bigi} points were {operation} from {n_points} '
                     f'to {edge_n_vertices} using 2nd order spline interpolation.')

        # calculate the length as if the edge was a straight line a.k.a. distance between junctions
        chordlen = euclidean_dist(points[0], points[-1])
        # calculate the arc-length as the sum of all small-edge lengths
        arclen = sum([euclidean_dist(points[i], points[i + 1]) for i in range(len(points) - 1)])
        # populate arrays
        nf = [arclen, chordlen, c1a, c2a, c1p, c2p]
        assert all((v and v > 0) for v in nf), f'Invalid values in node features: {nf}'
        node_features[bigi] = np.array(nf, dtype=np.float32)
        node_points[bigi] = points
        if tagged_nodes is not None:
            tagged_nodes[bigi] = is_tagged

        bigi += 1

    # TODO: for sure there are more efficient ways to exclude disconnected nodes. This works for now.
    adjacency = list()
    connected_set = set()
    for i, j in combinations(range(n_big_edges), 2):
        intersect = adj_tracker[i].intersection(adj_tracker[j])
        if intersect:
            assert len(intersect) == 1, f'A pair of big edges ({i} and {j}) share {len(intersect)} vertices (WTF?)'
            adjacency.append((i, j))
            connected_set.update([i, j])

    # indices of all connected nodes
    connected = sorted(connected_set)
    idx_old_to_new = {n: i for i, n in enumerate(connected)}  # indexer for node_features, ground_truth and forsys_preds

    # initialize adjacency matrix
    adj_mat = dok_matrix((len(connected),) * 2, dtype=bool)

    # edge features (store node indices as they get added to the adj. mat. so we can use them to order edge feats later)
    edge_index_buffer, edge_features_buffer = list(), list()

    # populate the adjacency matrix and edge features arrays
    for i, j in adjacency:
        # make sure that edge features for i->j are always  i> oooo[X]oooo >j, where [X] is the junction point (removed)
        if np.array_equal(node_points[i][0], node_points[j][0]):  # Xoooo Xoooo
            edge_points_ij = np.concatenate((node_points[i][::-1][:-1], node_points[j][1:]))
            joint_coords = node_points[i][0]
        elif np.array_equal(node_points[i][-1], node_points[j][0]):  # ooooX Xoooo
            edge_points_ij = np.concatenate((node_points[i][:-1], node_points[j][1:]))
            joint_coords = node_points[i][-1]
        elif np.array_equal(node_points[i][-1], node_points[j][-1]):  # ooooX ooooX
            edge_points_ij = np.concatenate((node_points[i][:-1], node_points[j][::-1][1:]))
            joint_coords = node_points[i][-1]
        elif np.array_equal(node_points[i][0], node_points[j][-1]):  # Xoooo ooooX
            edge_points_ij = np.concatenate((node_points[i][::-1][:-1], node_points[j][::-1][1:]))
            joint_coords = node_points[i][0]
        else:
            raise ValueError(f'WTF?\n{node_points[i]}\n{node_points[j]}')

        # move points so the joint ends up at the origin of coordinates.
        # the junction point is a constant feature (always [0, 0]), that's why we excluded it from the tensor beforehand
        edge_points_ij -= joint_coords
        # the features of the edge j->i are the sequence of points in the features of the edge i->j in reverse order
        edge_points_ji = edge_points_ij[::-1]
        # add edge features to the matrix
        edge_features_buffer.extend([edge_points_ij, edge_points_ji])

        # populate adj matrix and keep track of the rows/columns in order to sort edge features later
        i_, j_ = idx_old_to_new[i], idx_old_to_new[j]
        adj_mat[i_, j_] = True
        adj_mat[j_, i_] = True
        edge_index_buffer.extend([[i_, j_], [j_, i_]])

    if verbose:
        log.info('Normalizing data and running additional validations...')

    # filter out disconnected nodes from the original node_features matrix
    node_features = node_features[connected]
    if tagged_nodes is not None:
        tagged_nodes = tagged_nodes[connected]

    # order edge features in the way spektral expects them (as they appear in the adj. mat. sorted in row-major order)
    edge_index = np.array(edge_index_buffer)
    sort_idx = np.lexsort(np.flipud(edge_index.T))  # row-major order
    edge_features = np.array(edge_features_buffer, dtype=np.float32)[sort_idx]

    # rescale coordinates: divide by the maximum edge arc-length so all coords get rescaled to the range [-1, 1]
    max_length = node_features[:, 0].max()
    edge_features /= max_length

    # also scale lengths, cell areas and cell perimeters to the range [0, 1]
    node_features[:, 0:2] /= max_length  # chord-length and arc-length
    node_features[:, 2:4] /= node_features[:, 2:4].max()  # cell areas
    node_features[:, 4:6] /= node_features[:, 4:6].max()  # cell perimeters

    # convert adjacency matrix to CSR
    adj_mat = adj_mat.tocsr()

    # make sure that all nodes in the graph have at least one neighbor
    CN = ConnectedNodes(adj_mat)
    CN.assert_all_connected()

    assert edge_features.shape[0] == adj_mat.nnz, 'Expecting number of entries in the adj matrix to match n_edges'
    assert node_features.shape[0] == CN.get_count(), 'Expecting n_nodes to be equal to n_connected_nodes'
    assert (np.min(node_features) >= 0) and (np.max(node_features) <= 1), 'Node feats not in range [0, 1]'
    assert (np.min(edge_features) >= -1) and (np.max(edge_features) <= 1), 'Edge feats not in range [-1, 1]'

    # initialize the outputs
    out = {'a': adj_mat, 'x': node_features, 'e': edge_features}

    # return disconnected nodes if any
    n_removed = n_big_edges - len(connected_set)
    if n_removed > 0:
        log.warning(f'{n_removed} nodes were excluded for being disconnected from the graph.')
        out['removed_nodes'] = np.array(sorted(set(range(n_big_edges)) - connected_set), dtype=int)

    # return tagged nodes if any
    if (tagged_nodes is not None) and tagged_nodes.any():
        out['tagged_nodes'] = np.flatnonzero(tagged_nodes)

    # get a dataframe with predicted tensions and ground truths from forsys.frames.Frame object
    fs_frame_tensions = frame.get_tensions()

    if include_targets:
        assert np.allclose(ground_truth, fs_frame_tensions['gt'].values), 'ForSys ground-truths do not match ours.'
        # filter out tensions of disconnected nodes (we cannot predict them)
        targets = ground_truth[connected]
        assert targets.shape[0] == node_features.shape[0], 'Expecting n_targets to be equal to n_nodes'
        # rescale target vector to have mean=1
        targets_mean = targets.mean()
        if targets_mean > 0:  # targets mean can only be zero if all tensions are 0 (ground-truth not available)
            targets /= targets_mean
        out['y'] = targets

    if include_forsys_predictions:
        assert np.allclose(forsys_preds, fs_frame_tensions['stress'].values), 'ForSys tensions do not match ours.'
        # filter out predictions of disconnected nodes
        forsys_preds_f = forsys_preds[connected]
        assert forsys_preds_f.shape[0] == node_features.shape[0], 'Expecting n_forsys_preds to be equal to n_nodes'
        assert np.all(forsys_preds_f >= 0), f'Negative values in filtered forsys preds: {forsys_preds_f}'
        if np.any(forsys_preds_f == 0):
            log.warning(f'Found zeros in ForSys predictions which will be ignored in mean normalization. '
                        f'These values should be masked-out when calculating metrics to obtain accurate results.')
        # re-normalize forsys predictions after removing disconnected nodes (make sure to ignore zeros in this step)
        out['forsys_preds'] = forsys_preds_f / forsys_preds_f[forsys_preds_f > 0].mean()

    # end the timer, the rest is just for debugging purposes
    total_time = perf_counter() - st + load_time

    if return_timers:
        out.update({'total_time': total_time, 'load_time': load_time})
        if include_forsys_predictions:
            out['forsys_pred_time'] = forsys_pred_time

    # TODO: move this plotting step to its own function. Had to modify func in forsys (careful with future versions)
    if render_plots and (include_targets or include_forsys_predictions):
        if verbose:
            log.info('Plotting...')

        plots_base = Path(plots_dir)
        if include_targets:
            # plot with ground-truth tensions
            plot_with_force(frame, filename=str(plots_base / f'{plots_prefix}_gt'), force_to_plot='gt')

        if include_forsys_predictions:
            # plot with tensions predicted by forsys
            plot_with_force(frame, filename=str(plots_base / f'{plots_prefix}_forsys'), force_to_plot='tension')

    if return_frame:
        out['frame'] = frame

    return out


def se_output_to_graph(src_file: StrPath,
                       *,
                       include_targets: bool = True,
                       edge_n_vertices: int = 9,
                       apply_savgol_filter: bool = False,
                       include_forsys_predictions: bool = True,
                       forsys_solve_method: str = 'default',
                       render_plots: bool = False,
                       plots_dir: StrPath = './debug',
                       return_timers: bool = True,
                       tag_cell_interfaces: list | None = None,
                       jitter_kwargs: dict | None = None,
                       raise_if_gt_is_zero: bool = True,
                       debug_plots_prefix: str | None = None,
                       return_frame: bool = False,
                       verbose: bool = True
                       ) -> Dict[str, Any]:
    """Load a Surface Evolver dump, build a ForSys frame, and return graph arrays.

    Parameters
    ----------
    src_file
        Path to the Surface Evolver output file.
    include_targets
        Whether ground-truth tensions are present and should populate ``gt`` on the frame.
    edge_n_vertices
        Number of vertices sampled along each lattice edge for feature extraction.
    apply_savgol_filter
        If ``True``, smooth resampled vertex coordinates with a Savitzky–Golay filter.
    include_forsys_predictions
        Whether to run ForSys inference for auxiliary predictions in the output dict.
    forsys_solve_method
        Solver label passed to ForSys tension recovery.
    render_plots
        If ``True``, write debug plots under ``plots_dir``.
    plots_dir
        Directory for optional debug figures.
    return_timers
        Include timing fields in the returned dictionary.
    tag_cell_interfaces
        Optional list controlling interface tagging between cells.
    jitter_kwargs
        Optional dict of jitter parameters forwarded to vertex jittering.
    raise_if_gt_is_zero
        If ``True``, validate non-degenerate ground-truth tensions when present.
    debug_plots_prefix
        Filename prefix for plots; defaults to the stem of ``src_file``.
    return_frame
        If ``True``, include the constructed ``frame`` object under key ``'frame'``.
    verbose
        Enable logging of major steps.

    Returns
    -------
    dict
        Keys typically include ``'a'``, ``'x'``, ``'e'``, optional ``'y'``, timing keys,
        and optional ForSys prediction arrays depending on flags.
    """
    log.debug('Loading data in forsys...')

    src_path = Path(src_file)

    # start main timer
    st = perf_counter()

    # load data from SE output in forsys
    lattice = fs.surface_evolver.SurfaceEvolver(str(src_path))

    if jitter_kwargs:
        _jitter_vertices(lattice, **jitter_kwargs)

    frame = fs.frames.Frame(0, lattice.vertices, lattice.edges, lattice.cells, time=0, gt=include_targets)
    load_time = perf_counter() - st

    # define debug plot filenames prefixes from the name of the input file
    if debug_plots_prefix is None:
        debug_plots_prefix = src_path.stem if src_path.suffix else src_path.name

    # extract features and build the graph
    return _forsys_frame_to_graph(frame, include_targets, edge_n_vertices, apply_savgol_filter,
                                  include_forsys_predictions, forsys_solve_method, render_plots, plots_dir,
                                  return_timers, tag_cell_interfaces, raise_if_gt_is_zero, load_time,
                                  debug_plots_prefix, return_frame, verbose)


def skeleton_to_graph(src_file: StrPath,
                      gt_file: StrPath | None = None,
                      *,
                      mirror_y: bool = False,
                      edge_n_vertices: int = 9,
                      fixed_ne: int | None = None,
                      apply_savgol_filter: bool = False,
                      include_forsys_predictions: bool = True,
                      forsys_solve_method: str = 'default',
                      render_plots: bool = False,
                      plots_dir: StrPath = './debug',
                      return_timers: bool = True,
                      tag_cell_interfaces: list | None = None,
                      raise_if_gt_is_zero: bool = True,
                      debug_plots_prefix: str | None = None,
                       return_frame: bool = False,
                       verbose: bool = True
                       ) -> Dict[str, Any]:
    """Load a skeleton ``.tif``, build a mesh in ForSys, optionally attach myosin GT, and return graph tensors.

    Parameters
    ----------
    src_file
        Path to a binary skeleton image (``.tif``).
    gt_file
        Optional myosin intensity image (``.tif``) used to assign ground-truth tensions.
    mirror_y
        Passed to ForSys skeleton loading (flip image rows).
    edge_n_vertices
        Target number of vertices per edge for mesh generation.
    fixed_ne
        If set, overrides derived ``ne`` for ``generate_mesh``.
    apply_savgol_filter
        Whether to smooth resampled vertices with Savitzky–Golay filtering.
    include_forsys_predictions
        Whether to include ForSys baseline predictions in the output.
    forsys_solve_method
        ForSys solver label for tension recovery.
    render_plots
        Write debug plots when ``True``.
    plots_dir
        Output directory for debug plots.
    return_timers
        Attach timing metadata to the result dict.
    tag_cell_interfaces
        Optional interface tagging list for feature extraction.
    raise_if_gt_is_zero
        Validate non-zero ground truth when ``gt_file`` is provided.
    debug_plots_prefix
        Plot filename prefix; defaults to skeleton stem.
    return_frame
        Include the ForSys ``frame`` under ``'frame'`` when ``True``.
    verbose
        Enable progress logging.

    Returns
    -------
    dict
        Graph tensors and optional targets / predictions, same style as :func:`se_output_to_graph`.
    """
    if verbose:
        log.info('Loading data in forsys...')

    src_path = Path(src_file)
    # start main timer
    st = perf_counter()

    # load data from skeleton .tif file in forsys
    assert src_path.suffix.lower() == '.tif', 'Expecting a .tif file to generate the mesh.'
    ne = fixed_ne or (edge_n_vertices - 1)

    skeleton = fs.skeleton.Skeleton(str(src_path), mirror_y=mirror_y)
    vertices, edges, cells = skeleton.create_lattice()
    vertices, edges, cells, _ = fs.virtual_edges.generate_mesh(vertices, edges, cells, ne=ne)

    frame = fs.frames.Frame(0, vertices, edges, cells, time=0, gt=False)

    if gt_file is not None:
        gt_path = Path(gt_file)
        assert gt_path.suffix.lower() == '.tif', 'Expecting a .tif file to estimate the ground-truth from pixel intensities.'
        gt_tensions = fs.myosin.read_myosin(frame, str(gt_path), layers=2)  # layers value used in ForSys paper
        frame.assign_gt_tensions_to_big_edges(list(gt_tensions.values()))
    load_time = perf_counter() - st

    # define debug plot filenames prefixes from the name of the input file
    if debug_plots_prefix is None:
        debug_plots_prefix = src_path.stem if src_path.suffix else src_path.name

    include_targets = gt_file is not None

    # extract features and build the graph
    return _forsys_frame_to_graph(frame, include_targets, edge_n_vertices, apply_savgol_filter,
                                  include_forsys_predictions, forsys_solve_method, render_plots, plots_dir,
                                  return_timers, tag_cell_interfaces, raise_if_gt_is_zero, load_time,
                                  debug_plots_prefix, return_frame, verbose)
