"""Geometry helpers for the RIS CSI-feedback / beam-selection dataset (Step 1).

Conventions follow the DeepMIMO v4.0.5 source (``deepmimo/generator/geometry.py``,
``deepmimo/datasets/dataset.py``, ``deepmimo/generator/channel.py``):

* A panel ``shape=[My, Mz]`` lies in the local y-z plane; element index n = ny + nz*My.
* Element response: exp(+j*2*pi*d*(ny*u_y + nz*u_z)) with u_y = sin(th')sin(ph'),
  u_z = cos(th'), (th', ph') = zenith/azimuth in the *local* (rotated) frame, d in wavelengths.
* Channels are built as a_rx (x) a_tx with **no conjugate**, so for the cascaded channel
  c = g * (H w) both links add phase: c_n ~ exp(+j*2*pi*d*n.(u_ue + u_bs)).
* ``rotation=[alpha, beta, gamma]`` in degrees. With alpha = 0 the local frame is the
  global frame rotated by gamma about z (azimuth) and then tilted by beta, where beta > 0
  points the boresight (local +x) *below* the horizon. So ``[0, downtilt, azimuth]`` aims
  boresight at a target. (``Dataset.bs_look_at`` in 4.0.5 writes [az, el, roll] into the
  [x, y, z] slots, which does NOT aim the array; do not use it.)
"""

from __future__ import annotations

import numpy as np

C0 = 299_792_458.0
FC = 3.5e9
LAMBDA = C0 / FC

RIS_SHAPE = (36, 36)          # (My, Mz)
RIS_SPACING = 0.25            # wavelengths
BS_SHAPE = (3, 3)
BS_SPACING = 0.5
ANCHOR_ROWS = (4, 13, 22, 31)  # RISnet indices_sensors (0-based rows and columns)

D_MIN, D_MAX = 30.0, 120.0     # user distance window from the RIS [m]
UE_HEIGHT = 2.0


# --------------------------------------------------------------------------- arrays
def anchor_indices(rows=ANCHOR_ROWS, shape=RIS_SHAPE) -> np.ndarray:
    """Flat element indices (n = ny + nz*My) of the anchor elements, z-major order."""
    my = shape[0]
    return np.array([nz * my + ny for nz in rows for ny in rows], dtype=np.int64)


def aperture(n: int, spacing_wl: float) -> float:
    """Physical side length [m] of an n-element ULA with spacing in wavelengths."""
    return n * spacing_wl * LAMBDA


def far_field_distance(d_m: float) -> float:
    """Fraunhofer distance 2 D^2 / lambda."""
    return 2 * d_m**2 / LAMBDA


def dft_matrix(n: int, k: int | None = None) -> np.ndarray:
    """n x k DFT codebook, F[m, q] = exp(-j 2 pi m q / k) / sqrt(n). k = n*oversampling."""
    k = n if k is None else k
    m = np.arange(n)[:, None]
    q = np.arange(k)[None, :]
    return np.exp(-2j * np.pi * m * q / k) / np.sqrt(n)


def ris_codebook(shape=RIS_SHAPE, oversample: int = 1) -> np.ndarray:
    """2-D DFT codebook Psi = kron(F_z, F_y), rows match DeepMIMO element order.

    Column index = kz * Ky + ky.  Shape (My*Mz, Ky*Kz).
    """
    my, mz = shape
    fy = dft_matrix(my, my * oversample)
    fz = dft_matrix(mz, mz * oversample)
    return np.kron(fz, fy)


def predicted_bin(u: np.ndarray, n: int, spacing_wl: float, oversample: int = 1) -> np.ndarray:
    """DFT bin that co-phases exp(+j 2 pi d m u): q = round(K d u) mod K, K = n*oversample."""
    k = n * oversample
    return np.mod(np.rint(k * spacing_wl * np.asarray(u)), k).astype(np.int64)


def predicted_beam(u_y, u_z, shape=RIS_SHAPE, spacing_wl=RIS_SPACING, oversample=1):
    """Flat beam index kz*Ky + ky for direction sines (u_y, u_z)."""
    ky = predicted_bin(u_y, shape[0], spacing_wl, oversample)
    kz = predicted_bin(u_z, shape[1], spacing_wl, oversample)
    return kz * shape[0] * oversample + ky


# --------------------------------------------------------------------------- frames
def look_at_rotation(src, dst) -> np.ndarray:
    """DeepMIMO rotation [0, downtilt, azimuth] (deg) pointing boresight from src to dst."""
    d = np.asarray(dst, float) - np.asarray(src, float)
    az = np.degrees(np.arctan2(d[1], d[0]))
    tilt = np.degrees(np.arctan2(-d[2], np.hypot(d[0], d[1])))
    return np.array([0.0, tilt, az])


def to_local(vec: np.ndarray, rotation) -> np.ndarray:
    """Rotate global direction vectors (..., 3) into the panel frame of ``rotation`` (deg).

    Matches deepmimo.generator.geometry._rotate_angles for alpha = rotation[0] = 0.
    """
    rotation = np.asarray(rotation, float)
    if abs(rotation[0]) > 1e-9:
        raise NotImplementedError("only alpha = 0 (no roll about x) is supported")
    beta, gamma = np.radians(rotation[1]), np.radians(rotation[2])
    v = np.asarray(vec, float)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    cg, sg = np.cos(gamma), np.sin(gamma)
    x1, y1 = cg * x + sg * y, -sg * x + cg * y          # azimuth shift by -gamma
    cb, sb = np.cos(beta), np.sin(beta)
    return np.stack([cb * x1 - sb * z, y1, sb * x1 + cb * z], axis=-1)


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def direction_sines(src, dst, rotation) -> np.ndarray:
    """Local unit direction (x', y', z') = (boresight, u_y, u_z) from src to dst(s)."""
    return to_local(unit(np.asarray(dst, float) - np.asarray(src, float)), rotation)


# --------------------------------------------------------------------------- selection
def select_users(rx_pos, ris_pos, rotation=None, los=None, d_min=D_MIN, d_max=D_MAX):
    """Users in the distance window, with at least one path, in front of the panel."""
    rx_pos = np.asarray(rx_pos, float)
    dist = np.linalg.norm(rx_pos - np.asarray(ris_pos, float), axis=1)
    mask = (dist >= d_min) & (dist <= d_max)
    if los is not None:
        mask &= np.asarray(los) >= 0
    if rotation is not None:
        mask &= direction_sines(ris_pos, rx_pos, rotation)[:, 0] > 0
    return mask, dist


def aim_ris(rx_pos, ris_pos, los=None, d_min=D_MIN, d_max=D_MAX, n_iter=3):
    """Boresight toward the centroid of in-window users (iterated with the front mask)."""
    mask, _ = select_users(rx_pos, ris_pos, None, los, d_min, d_max)
    if not mask.any():
        return None, mask
    rot = look_at_rotation(ris_pos, rx_pos[mask].mean(0))
    for _ in range(n_iter):
        mask, _ = select_users(rx_pos, ris_pos, rot, los, d_min, d_max)
        if not mask.any():
            return None, mask
        rot = look_at_rotation(ris_pos, rx_pos[mask].mean(0))
    mask, _ = select_users(rx_pos, ris_pos, rot, los, d_min, d_max)
    return rot, mask


def label_stats(labels: np.ndarray, min_count: int = 50) -> dict:
    vals, cnt = np.unique(labels, return_counts=True)
    return {
        "n_distinct": int(len(vals)),
        "n_ge_min": int((cnt >= min_count).sum()),
        "users_in_ge_min": int(cnt[cnt >= min_count].sum()),
        "max_count": int(cnt.max()) if len(cnt) else 0,
    }


def predict_labels(ue_pos, ris_pos, bs_pos, rotation, oversample=1):
    """Geometric (dominant-path) best-beam prediction for each user."""
    u_ue = direction_sines(ris_pos, ue_pos, rotation)
    u_bs = direction_sines(ris_pos, np.asarray(bs_pos, float)[None], rotation)[0]
    u_sum = u_ue[:, 1:] + u_bs[1:]
    return predicted_beam(u_sum[:, 0], u_sum[:, 1], oversample=oversample), u_ue, u_bs


def decimate_grid(pos: np.ndarray, step_m: float) -> np.ndarray:
    """Boolean mask keeping grid points on a coarser lattice of pitch step_m (in x and y)."""
    if step_m <= 0:
        return np.ones(len(pos), bool)
    q = np.rint(pos[:, :2] / step_m).astype(np.int64)
    _, first = np.unique(q, axis=0, return_index=True)
    keep = np.zeros(len(pos), bool)
    keep[first] = True
    return keep
