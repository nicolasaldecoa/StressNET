"""StressNET: deep learning for mechanical stress inference in tissues.

This package exposes graph construction from ForSys/Surface Evolver inputs,
registered model weights (:func:`load_model`), inference helpers, and plotting
utilities. See the documentation for API details and the maintainer handover
page for publication setup (Read the Docs, Zenodo, PyPI).

Authors
-------
Primary author: Nicolás Aldecoa Rodrigo. Additional contributors: ``...``
(replace with names as appropriate).
"""

__version__ = '0.1.0'
__author__ = 'Nicolás Aldecoa Rodrigo, ...'
__license__ = 'BSD-3-Clause'

from .utils import data_utils
from .utils.io import save_graph_data, load_graph_data
from .utils.inference import predict, predict_augmented, calculate_metrics, frame_with_predicted_tensions
from .utils.plotting import plot_with_force
from .net.stressnet import get_model, build_stressnet
from .preprocessing import se_output_to_graph, skeleton_to_graph
from .models import list_models, load_model, get_model_path, clear_cache

__all__: list[str] = [
    'data_utils',
    'plot_with_force',
    'se_output_to_graph',
    'skeleton_to_graph',
    'save_graph_data',
    'load_graph_data',
    'predict',
    'predict_augmented',
    'calculate_metrics',
    'frame_with_predicted_tensions',
    'list_models',
    'load_model',
    'get_model_path',
    'clear_cache',
    'get_model',
    'build_stressnet',
]
