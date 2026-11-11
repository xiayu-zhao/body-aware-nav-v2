"""Shared panel helpers for the paper's figures.

The one rule worth a module: an image in a paper must never be scaled
anisotropically. A screenshot of a robot standing in a corridor stops
being a photograph of that scene the moment its aspect changes, and a
reader cannot tell a stretched render from a wide one. Where a panel's
shape does not match the image's, the image is *cropped* to the panel,
about its centre, which throws away content but never misrepresents it.
"""

from __future__ import annotations

import numpy as np


def axes_aspect(ax) -> float:
    """Width / height of an axes, in figure inches."""
    fig = ax.get_figure()
    box = ax.get_position()
    fw, fh = fig.get_size_inches()
    return (box.width * fw) / max(box.height * fh, 1e-9)


def crop_to_aspect(img: np.ndarray, aspect: float) -> np.ndarray:
    """Centre-crop an (H, W, ...) image to the given width/height ratio."""
    h, w = img.shape[0], img.shape[1]
    if h <= 0 or w <= 0 or aspect <= 0:
        return img
    want_w = aspect * h
    if want_w <= w:                      # too wide: trim left and right
        new_w = max(int(round(want_w)), 1)
        x0 = (w - new_w) // 2
        return img[:, x0:x0 + new_w]
    new_h = max(int(round(w / aspect)), 1)   # too tall: trim top and bottom
    y0 = (h - new_h) // 2
    return img[y0:y0 + new_h]


def show_fitted(ax, path, *, mode: str = "contain", max_crop: float = 0.18):
    """Draw an image at its true proportions.

    ``contain`` letterboxes it inside the panel: nothing is lost and
    nothing is distorted, at the price of some empty margin. ``cover``
    crops to fill instead, but only up to ``max_crop`` of the longer
    side -- past that a crop stops trimming margin and starts removing
    content, which on an annotated layout means removing the labels that
    make it worth showing.
    """
    import matplotlib.image as mpimg

    img = mpimg.imread(str(path))
    if mode == "cover":
        want = axes_aspect(ax)
        have = img.shape[1] / max(img.shape[0], 1)
        loss = 1.0 - min(want, have) / max(want, have)
        if loss <= max_crop:
            img = crop_to_aspect(img, want)
            ax.imshow(img, aspect="auto", interpolation="antialiased")
            return img
    # equal aspect letterboxes within the axes and centres the result
    ax.imshow(img, aspect="equal", interpolation="antialiased")
    return img


def margin_ring(ax, lo, hi, margin, *, color, lw=1.1, zorder=4):
    """The region a stated margin actually excludes, drawn as it is enforced.

    ``hazard_compliance`` measures Euclidean distance to the instance's
    bounding box, and ``margin_constraint`` forbids cells by a distance
    transform from that box. Both give the box grown by the margin in
    every direction -- straight along the sides, quarter-circle arcs at
    the corners. A plain rectangle drawn at ``lo - margin`` has square
    corners and claims an exclusion up to ``margin * (sqrt(2) - 1)``
    larger than the one enforced, which on a 2.5 m request is a metre of
    invented geometry at each corner.
    """
    import numpy as np
    from matplotlib.patches import FancyBboxPatch

    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    w, h = hi - lo
    patch = FancyBboxPatch(
        (lo[0], lo[1]), w, h,
        boxstyle=f"round,pad={float(margin)},rounding_size={float(margin)}",
        mutation_aspect=1.0, fill=False, edgecolor=color, lw=lw,
        linestyle=(0, (4, 3)), zorder=zorder)
    ax.add_patch(patch)
    return patch
