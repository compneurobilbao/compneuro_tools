"""
Spatial (voxelwise) mediation analysis with vectorized bootstrapping.

Model at each voxel v:
    M_v = a_v * X_v + e1                      (path a)
    Y_v = b_v * M_v + c'_v * X_v + e2         (paths b, c')
    Y_v = c_v * X_v + e3                      (path c, total effect)
    indirect effect: ab_v = a_v * b_v
    "_v" indicates that each path is estimated separately for each voxel.

Key design choices (see accompanying explanation):
  1. All voxels are fit simultaneously using closed-form OLS algebra
     (sums of squares / cross-products), not per-voxel regression calls.

  2. Bootstrap resampling is represented as an integer "case weight" per
     subject per bootstrap draw. This lets every bootstrap replicate,
     for every voxel, be computed via a handful of matrix multiplications
     (W @ X, W @ (X*M), ...) instead of a Python loop over 10,000 resamples.

  3. Voxels are processed in blocks so memory is bounded by
     (n_boot x block_size), never (n_boot x n_voxels).
"""

import numpy as np
from joblib import Parallel, delayed
from nilearn.maskers import NiftiMasker

from compneuro_tools._nifti import as_float32_img
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# 1. Bootstrap weights: generated ONCE, reused for every voxel and every path
# ---------------------------------------------------------------------------
def make_bootstrap_weights(n_subjects, n_boot, rng):
    """
    Returns W of shape (n_boot, n_subjects), float32.
    Row b gives, for bootstrap replicate b, how many times each subject
    was drawn (with replacement). Each row sums to n_subjects.

    Using this weight matrix in place of literal index-resampling is exact:
    all OLS sufficient statistics (sums, sums of squares, cross-products)
    are linear in the per-subject contributions, so a resampled dataset's
    sums equal the original subjects' values weighted by their draw counts.
    """
    idx = rng.integers(0, n_subjects, size=(n_boot, n_subjects))
    W = np.zeros((n_boot, n_subjects), dtype=np.float32)
    rows = np.repeat(np.arange(n_boot), n_subjects)
    np.add.at(W, (rows, idx.ravel()), 1)   # np.add.at accumulates duplicates correctly
    return W


# ---------------------------------------------------------------------------
# 2. Closed-form, fully vectorized mediation paths
#    Works for a block of voxels and a batch of weight-rows (bootstraps,
#    or a single all-ones row for the point estimate) simultaneously.
# ---------------------------------------------------------------------------
def mediation_paths(W, Xb, Mb, Yb, ridge=1e-8):
    """
    W  : (K, N)      case-weights; K = 1 for point estimate, K = n_boot for bootstrap
    Xb, Mb, Yb : (N, v) treatment / mediator / outcome values for a voxel block
                 (X, M can also be (N, 1) if treatment/mediator is a scalar
                 per subject rather than a map -- broadcasting handles it)
    Returns a, b, c, cprime, each shape (K, v)
    """
    n = Xb.shape[0]

    Sx, Sm, Sy = W @ Xb, W @ Mb, W @ Yb
    Sxx, Smm = W @ (Xb * Xb), W @ (Mb * Mb)
    Sxm, Sxy, Smy = W @ (Xb * Mb), W @ (Xb * Yb), W @ (Mb * Yb)

    S11 = Sxx - (Sx * Sx) / n
    S22 = Smm - (Sm * Sm) / n
    S12 = Sxm - (Sx * Sm) / n
    S1y = Sxy - (Sx * Sy) / n
    S2y = Smy - (Sm * Sy) / n

    a = S12 / (S11 + ridge)
    c = S1y / (S11 + ridge)

    denom = S11 * S22 - S12 ** 2
    denom = np.where(np.abs(denom) < ridge, ridge, denom)
    cprime = (S22 * S1y - S12 * S2y) / denom
    b = (S11 * S2y - S12 * S1y) / denom

    return a, b, c, cprime


def _compute_mediation_block(i, sl, Xb, Mb, Yb, W, W0, lo_pct, hi_pct):
    """Compute all summaries for a single voxel block."""
    a0, b0, c0, cp0 = mediation_paths(W0, Xb, Mb, Yb)
    a_boot, b_boot, _, _ = mediation_paths(W, Xb, Mb, Yb)
    ab_boot = a_boot * b_boot

    block_out = {
        "a_point": a0[0],
        "b_point": b0[0],
        "c_point": c0[0],
        "cprime_point": cp0[0],
        "ab_point": a0[0] * b0[0],
        "ab_se": ab_boot.std(axis=0, ddof=1),
        "ab_ci_lo": np.percentile(ab_boot, lo_pct, axis=0),
        "ab_ci_hi": np.percentile(ab_boot, hi_pct, axis=0),
    }

    p_pos = (ab_boot <= 0).mean(axis=0)
    p_neg = (ab_boot > 0).mean(axis=0)
    block_out["ab_p"] = 2 * np.minimum(p_pos, p_neg)

    return i, sl, block_out


# ---------------------------------------------------------------------------
# 3. Full pipeline: load -> block-wise bootstrap -> summary maps -> save
# ---------------------------------------------------------------------------
def run_spatial_mediation(
    x_paths, m_paths, y_paths, mask_img,
    n_boot=10_000, block_size=5_000, alpha=0.05,
    n_jobs=1,
    out_prefix="mediation", seed=0, save_raw_bootstrap_h5=None,
):
    rng = np.random.default_rng(seed)

    # ---- load & mask: 3D niftis -> (n_subjects, n_voxels) matrices ----
    masker = NiftiMasker(mask_img=mask_img, standardize=None)
    X = masker.fit_transform(x_paths).astype(np.float32)
    M = masker.transform(m_paths).astype(np.float32)
    Y = masker.transform(y_paths).astype(np.float32)
    n_subjects, n_voxels = X.shape
    print(f"n_subjects={n_subjects}, n_voxels={n_voxels}, n_boot={n_boot}")

    W = make_bootstrap_weights(n_subjects, n_boot, rng)      # (B, N), built once
    W0 = np.ones((1, n_subjects), dtype=np.float32)          # point-estimate "weights"

    lo_pct, hi_pct = 100 * alpha / 2, 100 * (1 - alpha / 2)

    out = {k: np.zeros(n_voxels, dtype=np.float32) for k in [
        "a_point", "b_point", "c_point", "cprime_point",
        "ab_point", "ab_se", "ab_ci_lo", "ab_ci_hi", "ab_p",
    ]}

    # optional: persistent storage of every raw bootstrap map (disk, chunked)
    h5_dset = None
    if save_raw_bootstrap_h5 is not None:
        import h5py
        h5_file = h5py.File(save_raw_bootstrap_h5, "w")
        h5_dset = h5_file.create_dataset(
            "ab_boot", shape=(n_boot, n_voxels), dtype="float32",
            chunks=(n_boot, min(block_size, n_voxels)), compression="gzip",
        )

    n_blocks = int(np.ceil(n_voxels / block_size))
    block_jobs = [
        (
            i,
            slice(i * block_size, min((i + 1) * block_size, n_voxels)),
        )
        for i in range(n_blocks)
    ]

    if h5_dset is not None or n_jobs == 1:
        for i, sl in block_jobs:
            Xb, Mb, Yb = X[:, sl], M[:, sl], Y[:, sl]

            # point estimate (no resampling)
            a0, b0, c0, cp0 = mediation_paths(W0, Xb, Mb, Yb)
            out["a_point"][sl] = a0[0]
            out["b_point"][sl] = b0[0]
            out["c_point"][sl] = c0[0]
            out["cprime_point"][sl] = cp0[0]
            out["ab_point"][sl] = a0[0] * b0[0]

            # full bootstrap distribution for this block only: (n_boot, block_size)
            a_boot, b_boot, _, _ = mediation_paths(W, Xb, Mb, Yb)
            ab_boot = a_boot * b_boot

            out["ab_se"][sl] = ab_boot.std(axis=0, ddof=1)
            out["ab_ci_lo"][sl] = np.percentile(ab_boot, lo_pct, axis=0)
            out["ab_ci_hi"][sl] = np.percentile(ab_boot, hi_pct, axis=0)

            # bootstrap two-sided p-value: proportion of resamples crossing zero
            p_pos = (ab_boot <= 0).mean(axis=0)
            p_neg = (ab_boot > 0).mean(axis=0)
            out["ab_p"][sl] = 2 * np.minimum(p_pos, p_neg)

            if h5_dset is not None:
                h5_dset[:, sl] = ab_boot

            del a_boot, b_boot, ab_boot  # free block memory before next iteration
            print(f"  block {i + 1}/{n_blocks} done")
    else:
        block_results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_compute_mediation_block)(
                i,
                sl,
                X[:, sl],
                M[:, sl],
                Y[:, sl],
                W,
                W0,
                lo_pct,
                hi_pct,
            )
            for i, sl in block_jobs
        )

        for i, sl, block_out in block_results:
            for name, values in block_out.items():
                out[name][sl] = values
            print(f"  block {i + 1}/{n_blocks} done")

    if h5_dset is not None:
        h5_file.close()

    # ---- multiple comparisons correction (voxelwise FDR, Benjamini-Hochberg) ----
    reject, p_fdr, _, _ = multipletests(out["ab_p"], alpha=alpha, method="fdr_bh")
    out["ab_p_fdr"] = p_fdr.astype(np.float32)
    out["ab_sig_fdr"] = reject.astype(np.float32)

    # ---- write each summary map back to brain space as NIfTI ----
    for name, vec in out.items():
        img = as_float32_img(masker.inverse_transform(vec[np.newaxis, :]))
        img.to_filename(f"{out_prefix}_{name}.nii.gz")

    return out, masker


if __name__ == "__main__":
    # Example call -- replace with your real file lists / mask
    # x_paths = [...]; m_paths = [...]; y_paths = [...]
    # run_spatial_mediation(x_paths, m_paths, y_paths, "mask.nii.gz", n_boot=10_000)
    import os

    x_paths = os.listdir("/home/teitxe/random/mediation_demo/X_maps_amyloid_PET")
    m_paths = os.listdir("/home/teitxe/random/mediation_demo/M_maps_tau_PET")
    y_paths = os.listdir("/home/teitxe/random/mediation_demo/Y_maps_tau_deviation")
    mask_path = "/home/teitxe/random/mediation_demo/HCP1112_GM_mask.nii.gz"

    run_spatial_mediation(
        x_paths=[os.path.join("/home/teitxe/random/mediation_demo/X_maps_amyloid_PET", f) for f in x_paths if "nii.gz" in f],
        m_paths=[os.path.join("/home/teitxe/random/mediation_demo/M_maps_tau_PET", f) for f in m_paths if "nii.gz" in f],
        y_paths=[os.path.join("/home/teitxe/random/mediation_demo/Y_maps_tau_deviation", f) for f in y_paths if "nii.gz" in f],
        mask_img=mask_path,
        n_boot=10_000,
        block_size=500,
        alpha=0.05,
        out_prefix="mediation",
        seed=0,
    )