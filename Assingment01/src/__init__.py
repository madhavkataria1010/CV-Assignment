"""
src — Modular panorama stitching package.

Submodules
----------
features    : SIFT feature extraction and matching
homography  : Homography estimation with RANSAC
warping     : Image warping, canvas computation, cylindrical projection
blending    : Naive overlay, linear feathering, multi-band (Laplacian) blending
utils       : Image I/O, gain compensation, cropping, comparison generation
superglue   : (Optional) SuperPoint + SuperGlue deep-learning matcher
"""

from src.features import extract_sift_features, match_features
from src.homography import compute_homography
from src.warping import (
    compute_canvas_size,
    warp_image,
    cylindrical_warp_vectorized,
    trim_black_borders,
)
from src.blending import (
    naive_stitch,
    linear_blend,
    multiband_blend_all,
)
from src.utils import (
    load_images,
    gain_compensate,
    crop_black,
    generate_comparison,
)

__all__ = [
    "extract_sift_features",
    "match_features",
    "compute_homography",
    "compute_canvas_size",
    "warp_image",
    "cylindrical_warp_vectorized",
    "trim_black_borders",
    "naive_stitch",
    "linear_blend",
    "multiband_blend_all",
    "load_images",
    "gain_compensate",
    "crop_black",
    "generate_comparison",
]
