# Graph Cut Image Segmentation Assignment

This repository contains a complete graph-cut-based image segmentation pipeline for foreground-background separation using user-guided annotations. The implementation explicitly models unary and pairwise costs, solves the binary labeling problem with min-cut / max-flow, performs iterative appearance refinement, mitigates artifacts, and exports report-ready visualizations and metrics.

## Project Structure

```text
.
├── configs/
│   ├── dataset.yaml
│   └── experiment.yaml
├── data/
│   ├── annotations/
│   └── input/
├── report/
│   ├── figures/
│   ├── references.bib
│   └── report.tex
├── results/
├── src/
└── tests/
```

## Environment

- Python `3.11`
- `uv` for environment management

## Setup

```bash
uv python install 3.11
uv venv --python 3.11 .venv
source .venv/bin/activate
uv sync --python .venv/bin/python --all-groups
```

If you prefer explicit package installation, install dependencies directly:

```bash
uv pip install --python .venv/bin/python imageio matplotlib numpy opencv-python PyMaxflow PyYAML scikit-image scipy pytest
```

## Usage

### 1. Bundle the deterministic sample dataset

```bash
python -m src.cli prepare-data
```

This creates three bundled sample images and their foreground/background scribble masks in `data/`.

### 2. Run the full segmentation pipeline

```bash
python -m src.cli run --config configs/experiment.yaml
```

### 3. Regenerate summary metrics from saved case outputs

```bash
python -m src.cli evaluate --results-dir results
```

### 4. Perform the complete workflow in one command

```bash
python -m src.cli all --config configs/experiment.yaml
```

## What the Pipeline Produces

For each input image, the pipeline saves the following to `results/<case_name>/`:

- `original.png`
- `annotation_overlay.png`
- `naive_mask.png`
- `graph_cut_raw_mask.png`
- `graph_cut_refined_mask.png`
- `final_overlay.png`
- `comparison_panel.png`
- `boundary_refinement.png`
- `energy_iterations.png`
- `metrics.json`

Aggregate outputs are written to `results/summary/`:

- `metrics.csv`
- `aggregate.json`

Report-ready figures are synchronized to `report/figures/`.

## Method Summary

- **Graph construction:** each pixel is a node with terminal edges to the source (foreground) and sink (background), plus 8-neighborhood pairwise edges.
- **Unary costs:** histogram-based foreground/background likelihoods are estimated in Lab color space from the user-provided scribble seeds and converted to negative log-likelihood costs.
- **Pairwise costs:** contrast-sensitive smoothness weights penalize label discontinuities while respecting strong image edges.
- **Optimization:** graph cut is solved with `PyMaxflow`, then the appearance models are updated iteratively from the current segmentation while preserving user seeds.
- **Artifact mitigation:** small isolated regions are removed, holes are filled, boundaries are smoothed, and seed-connected components are preserved.
- **Comparison:** the naive baseline uses the same unary model and annotations but omits pairwise smoothness and graph optimization.

## Reproducibility Notes

- The dataset is bundled and deterministic.
- The experiments are controlled through YAML configuration files in `configs/`.
- The report source is stored in `report/report.tex` with supporting figures in `report/figures/`.
- The main experiments use sparse foreground/background scribbles. Bounding boxes are supported as weak initialization constraints through the dataset configuration.

## Testing

Run the full test suite with:

```bash
pytest
```

The tests cover:

- histogram-based foreground/background modeling
- graph-cut solver behavior on toy inputs
- refinement behavior and seed preservation
- end-to-end smoke execution on a bundled sample image

## Report Source

The LaTeX source for the submission report is located in `report/report.tex`. Upload the `report/` directory to Overleaf together with the generated figures in `report/figures/`.
