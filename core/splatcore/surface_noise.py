"""Conservative surface-haze candidates, not a reconstruction or denoiser.

Fit local planes to opaque reference splats; score faint splats displaced
from those planes. Edges, lines, rough neighbourhoods and distant regions
are rejected. Reference sampling and batched queries bound temporary memory.
Results are for user review: opacity and geometry cannot prove a splat is noise.
"""
import numpy as np
from scipy.spatial import cKDTree


def analyse(s, candidates=None, reference_limit=250000, alive=None):
    """Return document-local IDs and confidence, without copying radiance data.

    ``s`` only needs xyz, opacity and scale arrays. Deleted splats must be
    excluded from both candidates and reference planes via ``alive``.
    """
    if not isinstance(reference_limit, (int, np.integer)) or reference_limit < 24:
        raise ValueError('Surface haze needs a reference limit of at least 24.')
    n = len(s.xyz)
    finite = np.isfinite(s.xyz).all(1) & np.isfinite(s.opacity) & np.isfinite(s.scale).all(1)
    valid = finite.copy()
    if alive is not None:
        if np.shape(alive) != (n,):
            raise ValueError('The surface haze visibility mask does not match the splats.')
        valid &= np.asarray(alive, dtype=bool)
    # Float32 is sufficient for a confidence score; clipping avoids overflow
    # warnings for valid extreme logits and halves the all-splat alpha buffer.
    alpha = 1 / (1 + np.exp(-np.clip(s.opacity, -80, 80).astype(np.float32)))
    if candidates is None:
        ids = np.flatnonzero(valid & (alpha < .6))
    else:
        ids = np.asarray(candidates, dtype=np.int64).reshape(-1)
        ids = np.unique(ids[(ids >= 0) & (ids < n)])
        ids = ids[valid[ids] & (alpha[ids] < .6)]
    if not len(ids):
        return ids, np.zeros(0, np.float32), {'references': 0, 'tested': 0,
                                            'invalidSkipped': int((~finite).sum())}
    reference = np.flatnonzero(valid & (alpha >= .65))
    if len(reference) < 24:
        raise ValueError('Not enough opaque surface splats to estimate surface haze.')
    if len(reference) > reference_limit:
        reference = np.sort(np.random.default_rng(0).choice(reference, reference_limit, replace=False))
    xyz = s.xyz[reference].astype(np.float64)
    tree = cKDTree(xyz)
    scores = np.zeros(len(ids), np.float32)
    # Log space avoids overflowing unusually large splat scales.
    ref_size = np.max(s.scale[reference], axis=1)
    for start in range(0, len(ids), 8192):
        chunk = ids[start:start + 8192]
        pos = s.xyz[chunk].astype(np.float64)
        distances, neighbours = tree.query(pos, k=24, workers=-1)
        patches = xyz[neighbours]
        centre = patches.mean(1)
        centred = patches - centre[:, None, :]
        covariance = np.einsum('nki,nkj->nij', centred, centred) / 24
        values, vectors = np.linalg.eigh(covariance)
        values = np.maximum(values, 0)
        normal = vectors[:, :, 0]
        radius = np.sqrt(values[:, 2])
        thickness = np.sqrt(values[:, 0])
        residual = np.abs(np.einsum('ni,ni->n', pos - centre, normal))
        # Need a locally flat two-dimensional patch, not a line or crease.
        supported = (values[:, 1] > values[:, 2] * .15) & (values[:, 0] < values[:, 1] * .08)
        supported &= (radius > 1e-9) & (distances[:, 0] < radius * 3)
        separation = residual / np.maximum(np.maximum(thickness * 3, radius * .2), 1e-9)
        off_surface = np.clip((separation - 1) / 3, 0, 1)
        faint = np.clip((.6 - alpha[chunk]) / .5, 0, 1)
        large = np.clip((np.max(s.scale[chunk], axis=1) - np.median(ref_size[neighbours], axis=1)
                         - np.log(2)) / np.log(4), 0, 1)
        scores[start:start + len(chunk)] = supported * off_surface * faint * (.7 + .3 * large)
    return ids, scores, {'references': len(reference), 'tested': len(ids),
                        'invalidSkipped': int((~finite).sum())}
