# StressNET User Agent Guide

This file is for **users of StressNET** who want to ask an AI assistant
(Cursor, GitHub Copilot Chat, Claude Code, ChatGPT, etc.) to help write scripts,
load data, run inference, create plots, or adapt the example notebooks.

If you are using an assistant, attach this file and say:

> Read `AGENTS.md` first. I want to use StressNET on my data. Ask me for the
> input file paths and write a script/notebook cell using the public API only.

## What StressNET Does

StressNET infers mechanical stress/tension values on tissue interfaces from
graph representations of cell boundaries. Typical user workflows are:

1. Convert input data into a graph dictionary.
2. Load a registered StressNET model.
3. Run inference.
4. Compare against ground truth or ForSys predictions when available.
5. Visualize inferred stresses on a ForSys frame.

## What the Assistant Should Ask the User

Before writing code, ask the user:

- Are you using **Surface Evolver output** (`.dmp` / simulation-style input) or
  **experimental skeleton/myosin images** (`.tif`)?
- What are the paths to the input files?
- Do you have ground-truth myosin/tension data?
- Do you want the default pretrained model or a specific registered model?
- Do you want plots saved to files or shown interactively?
- Are you running from the cloned repository examples, or from an installed
  Python package?

## Installation Assumptions for User Scripts

StressNET currently targets **Python 3.10** because of the TensorFlow 2.8 stack.

For a simple user script, assume:

```bash
pip install stressnet
```

For the repository examples, ask where the notebook or script is running:

- If the working directory is the repository root, use paths like
  `examples/example_data/...`.
- If the working directory is the `examples/` folder (as in the shipped
  notebooks), use paths like `example_data/...` and import
  `download_radial_1d_2d_inverted_9v` as a sibling module.

## Working Directory Assumptions

The shipped notebooks in `examples/` assume Jupyter is started with the
`examples/` folder as the working directory. Their paths use `Path('example_data') / ...`,
not `Path('examples/example_data') / ...`.

Before writing paths in a script or notebook cell, ask the user:

- Are you running from the repository root or from `examples/`?
- Are you using a cloned repository checkout or an installed `stressnet` package
  plus a separate copy of the example notebooks?

When adapting repository examples, preserve the same working-directory convention
as the source notebook unless the user explicitly wants repository-root paths.

## Example Notebooks

These are the canonical workflows in `examples/`:

| Notebook | Input type | Key steps |
| --- | --- | --- |
| `inference_in_silico.ipynb` | Surface Evolver `.dmp` | `list_models()` → `load_model()` → `se_output_to_graph(..., include_forsys_predictions=True, return_frame=True)` → `predict()` / `predict_augmented()` → `calculate_metrics()` → `frame_with_predicted_tensions()` → `plot_with_force()` |
| `inference_in_vivo.ipynb` | Skeleton/myosin `.tif` | `load_model()` → `skeleton_to_graph(..., fixed_ne=6, gt_file=..., include_forsys_predictions=True, return_frame=True)` → compare default and `STRESSNET-FINETUNED-MYOSIN-88-A` models → metrics and stress plots |
| `finetuning.ipynb` | Downloaded `.npz` graph files | `ensure_dataset_downloaded()` → `split_data()` → `GraphDataGenerator` → `load_model(..., fine_tune_layers=[...])` → Keras training with callbacks |

Notebook-specific notes:

- **In silico:** example file is
  `example_data/inference/synthetic/normal_furrow_circular/step_24.dmp` when cwd is
  `examples/`. The notebook compares single predictions, augmented predictions,
  ForSys baselines, and ground truth when `data["y"]` is present.
- **In vivo:** example files are under
  `example_data/inference/experimental/xenopus_myosin/` (`skeleton.tif`,
  `myosin.tif`). Use `fixed_ne=6` as in the notebook. The finetuned myosin model
  is `STRESSNET-FINETUNED-MYOSIN-88-A`.
- **Finetuning:** the dataset is not tracked in git. Call
  `ensure_dataset_downloaded()` first. Default finetuning unfreezes
  `reg_head_fc_2`, `reg_head_ln_2`, and `reg_head_out`.

Saved notebook outputs:

- The example notebooks may contain saved execution outputs and embedded plot
  images. That is fine for demos, but assistants should not assume saved outputs
  are always present or up to date. Prefer generating fresh outputs when helping
  users reproduce a workflow.

## Public API Surface

The top-level `stressnet` package re-exports the user-facing API:

- Preprocessing:
  - `stressnet.se_output_to_graph`
  - `stressnet.skeleton_to_graph`
- Model registry / loading:
  - `stressnet.list_models`
  - `stressnet.load_model`
  - `stressnet.get_model_path`
  - `stressnet.clear_cache`
  - `stressnet.build_stressnet`
  - `stressnet.get_model`
- Inference:
  - `stressnet.predict`
  - `stressnet.predict_augmented`
  - `stressnet.calculate_metrics`
  - `stressnet.frame_with_predicted_tensions`
- Plotting and IO:
  - `stressnet.plot_with_force`
  - `stressnet.save_graph_data`
  - `stressnet.load_graph_data`
- Data utilities:
  - `stressnet.data_utils`

Prefer using these public entry points in examples and notebooks.

## Model Registry

Registered model names used by examples:

- `STRESSNET-PRETRAINED-A`
- `STRESSNET-PRETRAINED-B`
- `STRESSNET-PRETRAINED-C`
- `STRESSNET-PRETRAINED-D`
- `STRESSNET-PRETRAINED-E`
- `STRESSNET-FINETUNED-MYOSIN-88-A`

Use `stressnet.load_model()` for the default pretrained model and
`stressnet.load_model("<MODEL_NAME>")` for a specific registered model.

Useful registry calls:

```python
import stressnet

print(stressnet.list_models())
weights_path = stressnet.get_model_path("STRESSNET-PRETRAINED-A")
stressnet.clear_cache("STRESSNET-PRETRAINED-A")  # remove one cached weights file
stressnet.clear_cache()  # remove all cached StressNET model weights
```

## User Workflows

### Inference on Synthetic / Surface Evolver Data

Use this when the user has a Surface Evolver / simulation file.

```python
from pathlib import Path

import stressnet

src_file = Path("examples/example_data/inference/synthetic/normal_furrow_circular/step_24.dmp")

data = stressnet.se_output_to_graph(
    src_file=src_file,
    include_forsys_predictions=True,
    return_frame=True,
)

model = stressnet.load_model()
preds = stressnet.predict(model, data)
aug_preds = stressnet.predict_augmented(model, data, n_augmentations=9, seed=1337)

if "y" in data:
    metrics = stressnet.calculate_metrics(data["y"], aug_preds)
```

Expected graph dictionary keys commonly include:

- `a`: sparse adjacency matrix
- `x`: node features
- `e`: edge features, shaped like `(n_edges, n_edge_vertices - 1, 2)`
- `y`: targets when available
- `forsys_preds`: ForSys baseline predictions when requested
- `frame`: ForSys frame when `return_frame=True`

### Inference on Experimental Skeleton / Myosin Data

Use this when the user has a skeleton image and optional myosin image.

```python
from pathlib import Path

import stressnet

skeleton_file = Path("examples/example_data/inference/experimental/xenopus_myosin/skeleton.tif")
myosin_file = Path("examples/example_data/inference/experimental/xenopus_myosin/myosin.tif")

data = stressnet.skeleton_to_graph(
    src_file=skeleton_file,
    gt_file=myosin_file,
    fixed_ne=6,
    include_forsys_predictions=True,
    return_frame=True,
)

model = stressnet.load_model()
preds = stressnet.predict(model, data)
```

For predicted tension visualization:

```python
frame_with_preds = stressnet.frame_with_predicted_tensions(data["frame"], preds)
stressnet.plot_with_force(frame_with_preds, force_to_plot="stress")
```

`frame_with_predicted_tensions` must not mutate the original frame. It deep-copies
the frame and assigns one predicted tension per big edge to all underlying small
edges.

### Finetuning Example Data

The finetuning dataset is intentionally not tracked in git. The helper script
downloads a zip and validates that these directories exist:

- `examples/example_data/finetuning/radial_1d_2d_inverted_9v/train`
- `examples/example_data/finetuning/radial_1d_2d_inverted_9v/test`

Notebook usage:

```python
# With the working directory set to the notebook's folder (`examples/`), the downloader
# is a sibling module:
from download_radial_1d_2d_inverted_9v import ensure_dataset_downloaded

ensure_dataset_downloaded()
```

Finetuning workflow uses:

- `stressnet.data_utils.split_data`
- `stressnet.data_utils.GraphDataGenerator`
- `stressnet.data_utils.EdgeFeaturesRandomRotation`
- `stressnet.get_model_path("STRESSNET-PRETRAINED-A")`
- `stressnet.build_stressnet(..., fine_tune_layers=[...])`

Do not commit downloaded `.npz` finetuning data.

## Public API Cookbook

### Save and Reload Graph Data

Use these functions when the user wants to preprocess once and reuse graph data
later without rerunning ForSys preprocessing.

```python
import stressnet

data = stressnet.se_output_to_graph(
    src_file="path/to/input.dmp",
    include_forsys_predictions=True,
)

stressnet.save_graph_data(
    "graph_data/example_graph.npz",
    adj_mat=data["a"],
    node_features=data["x"],
    edge_features=data["e"],
    targets=data.get("y"),
    forsys_predictions=data.get("forsys_preds"),
    compressed=True,
)

loaded = stressnet.load_graph_data("graph_data/example_graph.npz")
preds = stressnet.predict(stressnet.load_model(), loaded)
```

### Build a Model Manually

Most users should use `stressnet.load_model()`. Use `get_model_path` and
`build_stressnet` only when the user wants to fine-tune or customize architecture
arguments.

```python
import stressnet

weights_path = stressnet.get_model_path("STRESSNET-PRETRAINED-A")
model = stressnet.build_stressnet(
    load_weights_path=str(weights_path),
    edge_n_vertices=9,
    fine_tune_layers=["reg_head_fc_2", "reg_head_ln_2", "reg_head_out"],
)
```

Use `stressnet.get_model(weights_path=..., points_per_edge=9)` if the user has a
weights file and wants the standard ready-to-use model constructor.

### Split Finetuning Files

Use `stressnet.data_utils.split_data` on a Pandas `Series` of `.npz` paths.

```python
from pathlib import Path

import pandas as pd
import stressnet

data_dir = Path("examples/example_data/finetuning/radial_1d_2d_inverted_9v")
train_files = pd.Series(sorted(str(path) for path in (data_dir / "train").glob("*.npz")))

sources = stressnet.data_utils.split_data(
    train_files,
    val_split=0.25,
    test_split=0,
    seed=1337,
)

print(sources.n_train, sources.n_val, sources.n_test)
```

### Train or Evaluate with GraphDataGenerator

Use `GraphDataGenerator` for one graph per step. It loads `.npz` graph files
saved with `save_graph_data` and returns Keras-ready inputs.

```python
from pathlib import Path

import pandas as pd
import stressnet

data_dir = Path("examples/example_data/finetuning/radial_1d_2d_inverted_9v/train")
paths = pd.Series(sorted(path.name for path in data_dir.glob("*.npz")))

generator = stressnet.data_utils.GraphDataGenerator(
    paths,
    basedir=str(data_dir),
    e_augmentations=[
        stressnet.data_utils.EdgeFeaturesRandomRotation(always_apply=True),
    ],
    random_seed=1337,
)

inputs, targets = generator[0]
```

Use `DisjointModeDataGenerator` when batching several graphs in Spektral disjoint
mode:

```python
generator = stressnet.data_utils.DisjointModeDataGenerator(
    paths,
    batch_size=8,
    basedir=str(data_dir),
    random_seed=1337,
)
```

### Edge Feature Augmentation Utilities

These are useful for data augmentation and debugging.

```python
import numpy as np
import stressnet

edge_features = np.zeros((4, 8, 2), dtype=np.float32)
rotated = stressnet.data_utils.rotate_2d_batch(edge_features, angle_rad=np.pi / 2)

rotation_aug = stressnet.data_utils.EdgeFeaturesRandomRotation(always_apply=True)
jitter_aug = stressnet.data_utils.EdgeFeaturesRandomJitter(jitter_e=0.05, always_apply=True)
```

### Connected Node Checks

Use `ConnectedNodes` when the user needs to verify that all graph nodes have at
least one connection.

```python
import stressnet

connected = stressnet.data_utils.ConnectedNodes(data["a"])
connected.assert_all_connected()
disconnected_indices = connected.get_indices("disconnected")
```

### Resampling Utilities

Use these when manipulating edge coordinate samples directly.

```python
import numpy as np
import stressnet

vertices = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
resampled = stressnet.data_utils.resample_vertices(vertices, target_length=9)

with_nans = np.array([[0.0, 0.0], [np.nan, np.nan], [2.0, 0.0]])
filled = stressnet.data_utils.interpolate_vertices_fill_nans(with_nans)
```

### Loading File Lists from Logs

Use this when simulation logs contain a `simulation` column and an `n_nodes`
column.

```python
import stressnet

paths = stressnet.data_utils.load_npz_paths_from_logs("logs.csv")
sources = stressnet.data_utils.load_split_merge_from_logs(
    ["logs_a.csv", "logs_b.csv"],
    val_split=0.2,
    test_split=0.2,
    seed=1337,
)
```

### Logging Header Helper

Use `get_logs_headers` when creating tabular logs that combine base metrics and
parameter settings.

```python
headers = stressnet.data_utils.get_logs_headers(
    {"learning_rate": 5e-4, "batch_size": 8},
    include_forsys_predictions=True,
)
```

## Common User Requests and How to Answer

### "Load my data and run the default model"

Ask for:

- input type: Surface Evolver file or skeleton/myosin images
- file paths
- output directory for plots or predictions

Then write a script using `se_output_to_graph` or `skeleton_to_graph`, followed
by `load_model()` and `predict()`.

### "Compare StressNET against ForSys"

Use preprocessing with `include_forsys_predictions=True`. If targets exist,
compare:

```python
stressnet.calculate_metrics(data["y"], stressnet_preds)
stressnet.calculate_metrics(data["y"], data["forsys_preds"])
```

Only compute metrics when `data["y"]` exists.

### "Plot predicted stresses"

Use:

```python
frame_with_preds = stressnet.frame_with_predicted_tensions(data["frame"], preds)
stressnet.plot_with_force(
    frame_with_preds,
    filename="stressnet_prediction.png",
    force_to_plot="stress",
    cbar=True,
)
```

If no `frame` is present, the user needs to rerun preprocessing with
`return_frame=True`.

### "Use the finetuned myosin model"

Use:

```python
model = stressnet.load_model("STRESSNET-FINETUNED-MYOSIN-88-A")
```

This is intended for myosin-style experimental examples.

## Plotting Contracts

`stressnet.plot_with_force(frame, force_to_plot=...)` accepts:

- `None`: discrete colors per internal big edge
- `"stress"` or `"tension"`: continuous edge `tension`
- `"gt"` or `"ground-truth"`: continuous edge `gt`

External edges are plotted in black.

## Things Assistants Should Avoid for End Users

- Do not edit StressNET source code unless the user explicitly asks to modify the library.
- Do not assume the user's paths. Ask for exact file locations.
- Do not invent ground truth if `y` / myosin targets are unavailable.
- Do not save large downloaded datasets into git.
- Do not run long training jobs without confirming the expected runtime and hardware.

## Minimal Script Template

```python
from pathlib import Path

import numpy as np
import stressnet


def run_stressnet_on_surface_evolver(src_file: str, output_png: str | None = None):
    data = stressnet.se_output_to_graph(
        src_file=Path(src_file),
        include_forsys_predictions=True,
        return_frame=True,
    )
    model = stressnet.load_model()
    preds = stressnet.predict(model, data)

    if output_png is not None:
        frame_with_preds = stressnet.frame_with_predicted_tensions(data["frame"], preds)
        stressnet.plot_with_force(
            frame_with_preds,
            filename=output_png,
            force_to_plot="stress",
            cbar=True,
        )

    if "y" in data:
        r, r2, mape, score = stressnet.calculate_metrics(data["y"], preds)
        print(f"r={r:.3f}, r2={r2:.3f}, MAPE={mape:.2f}%, score={score:.2f}")

    return np.asarray(preds)
```
