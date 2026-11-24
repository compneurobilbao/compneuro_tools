"""Small NIfTI helpers shared by the CLI tools."""
import nibabel as nib
import numpy as np


def as_float32_img(img):
    """Return a copy of ``img`` whose header stores float32 data.

    ``NiftiMasker.inverse_transform`` / ``new_img_like`` copy the header of the
    reference image in some nilearn versions (e.g. 0.13). If that reference is
    a uint8 mask, nibabel then writes continuous maps (betas, T, Z, p,
    residuals) as scaled uint8, i.e. with only ~256 distinct values.
    Forcing float32 avoids this quantisation regardless of the nilearn version.
    """
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    header = img.header.copy()
    header.set_data_dtype(np.float32)
    header.set_slope_inter(None, None)  # no scaling for float data
    return nib.Nifti1Image(data, img.affine, header)
