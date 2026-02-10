# Panorama Stitching — CV Assignment 1

Stitch 3+ overlapping images into a seamless panorama using SIFT features, RANSAC homography estimation, and advanced blending techniques.

## Features

- **SIFT** keypoint detection (up to 5 000 features per image)
- **FLANN** approximate nearest-neighbour matching with Lowe's ratio test
- **RANSAC** robust homography estimation with centre-referenced alignment
- **Blending methods**: naive overlay, linear feathering, multi-band (Laplacian pyramid)
- **Cylindrical warping** for wide-angle scenes
- Gain compensation for exposure equalisation

## Project Structure

```
.
├── panorama_stitcher.py   # CLI entry point
├── src/                   # Core library
│   ├── config.py          # Tuneable constants
│   ├── features.py        # SIFT extraction & FLANN matching
│   ├── homography.py      # Homography estimation (DLT + RANSAC)
│   ├── warping.py         # Perspective & cylindrical warping
│   ├── blending.py        # Naive, linear, multi-band blending
│   └── utils.py           # I/O, gain compensation, cropping
├── images/                # Input images (overlapping photos)
└── output/                # Generated panoramas & visualisations
```

## Setup

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (fast Python package manager)

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create environment & install dependencies

```bash
uv venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows

uv pip install opencv-contrib-python numpy matplotlib
```

## Usage

Place 3 or more overlapping images (sorted left-to-right) in `images/`, then run:

```bash
python panorama_stitcher.py --images_dir images/ --method all
```

### CLI Options

| Flag | Default | Description |
|---|---|---|
| `--images_dir` | `images/` | Directory containing input images |
| `--output_dir` | `output/` | Directory for output files |
| `--method` | `all` | `standard`, `multiband`, `cylindrical`, `superglue`, or `all` |
| `--max_dim` | `1200` | Resize longest edge (0 = no resize) |
| `--no_crop` | — | Skip automatic black-border cropping |

### Examples

Run only multi-band blending:

```bash
python panorama_stitcher.py --method multiband
```

Run on high-res images without downscaling:

```bash
python panorama_stitcher.py --max_dim 0
```

## Output

The pipeline generates the following in `output/`:

| File | Description |
|---|---|
| `keypoints_*.jpg` | SIFT keypoint visualisations |
| `matches_*-*.jpg` | Feature match visualisations |
| `warped_*.jpg` | Perspective-warped images on common canvas |
| `naive_stitch.jpg` | Naive overlay (visible seams) |
| `panorama_linear.jpg` | Linear feathering blend |
| `panorama_multiband.jpg` | Multi-band Laplacian pyramid blend |
| `panorama_cylindrical.jpg` | Cylindrical projection panorama |
| `comparison.jpg` | Side-by-side comparison of all methods |

## Configuration

Edit `src/config.py` to tune parameters:

```python
SIFT_NFEATURES  = 5000   # Max keypoints per image
MIN_MATCH_COUNT = 10     # Min matches for homography
LOWE_RATIO      = 0.75   # Ratio test threshold
RANSAC_THRESH   = 5.0    # RANSAC reprojection threshold (px)
PYRAMID_LEVELS  = 6      # Laplacian pyramid depth
MAX_DIMENSION   = 1200   # Resize limit (longest edge)
```
