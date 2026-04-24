API reference
=============

Public symbols are defined on the :mod:`stressnet` package. Heavy runtime
dependencies (TensorFlow, Spektral, ForSys) may be mocked when building these
docs without a full scientific stack.

Top-level package
-----------------

.. automodule:: stressnet

Preprocessing
-------------

.. autofunction:: stressnet.se_output_to_graph

.. autofunction:: stressnet.skeleton_to_graph

Models
------

.. autofunction:: stressnet.list_models

.. autofunction:: stressnet.load_model

.. autofunction:: stressnet.get_model_path

.. autofunction:: stressnet.clear_cache

.. autofunction:: stressnet.get_model

.. autofunction:: stressnet.build_stressnet

Inference
---------

.. autofunction:: stressnet.predict

.. autofunction:: stressnet.predict_augmented

.. autofunction:: stressnet.calculate_metrics

.. autofunction:: stressnet.frame_with_predicted_tensions

Plotting
--------

.. autofunction:: stressnet.plot_with_force

Graph IO
--------

.. autofunction:: stressnet.save_graph_data

.. autofunction:: stressnet.load_graph_data
