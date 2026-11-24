import os
import subprocess as sp

from argparse import ArgumentParser
from glob import glob

import polars as pl
import numpy as np
from scipy.stats import norm


PVALS = [0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]


def _setup_parser():
    parser = ArgumentParser(description=("Correct GLM results using cluster correction"
                                         "based on MC simulations."))
    parser.add_argument(
        "--residuals",
        type=str,
        required=True,
        help="Path to the residuals image.",
    )

    parser.add_argument(
        "--zmap",
        type=str,
        required=True,
        help="Path to the Z-stat image. Assuming 2 sided test.",
    )

    parser.add_argument(
        "--mask",
        type=str,
        required=True,
        help="Path to the mask file."
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output directory.",
    )

    parser.add_argument(
        "--n_iterations",
        type=int,
        default=10_000,
        help="Number of iterations of MC simulation for cluster correction."
    )

    parser.add_argument(
        "--pthr",
        type=float,
        default=0.001,
        help=("Primary (cluster-forming) two-sided voxelwise p-threshold, chosen a priori. "
              f"Must be one of {PVALS}. The corrected map NN2_2sided_results_mask.nii.gz is built "
              "ONLY at this threshold (cluster-level FWE alpha = 0.05). Default: 0.001."),
    )

    parser.add_argument(
        "--save_all_pthr",
        action="store_true",
        default=False,
        help=("Also save the corrected clusters at every other p-threshold as separate "
              "acf_NN2_2sided_pval_<p>.nii.gz files (exploratory only: each one controls FWE "
              "on its own, but looking at several of them, or their union, does not)."),
    )

    parser.add_argument(
        "--recompute",
        action="store_true",
        default=False,
        help=("Re-estimate the ACF (3dFWHMx) and re-run 3dClustSim even if acf.txt / "
              "acf.NN2_2sided.1D already exist (e.g. after re-running fit_glm)."),
    )

    return parser


def _check_args(parser):
    args = parser.parse_args()

    # Check if the input file exists
    if not os.path.exists(args.residuals):
        raise FileNotFoundError(f"Input file {args.residuals} does not exist.")
    else:
        args.residuals = os.path.abspath(args.residuals)

    # Check if the Z-stat image exists
    if not os.path.exists(args.zmap):
        raise FileNotFoundError(f"Z-stat image {args.zmap} does not exist.")
    else:
        args.zmap = os.path.abspath(args.zmap)

    # Check if the mask file exists
    if not os.path.exists(args.mask):
        raise FileNotFoundError(f"Mask file {args.mask} does not exist.")
    else:
        args.mask = os.path.abspath(args.mask)

    # Check if the output directory is a directory
    if not os.path.isdir(args.output):
        raise NotADirectoryError(f"Output path {args.output} is not a directory.")
    else:
        args.output = os.path.join(os.path.abspath(args.output), "cluster_correction")
        if os.path.exists(args.output):
            print("[INFO] Output directory already exists. Overwriting!")
        else:
            os.makedirs(args.output, exist_ok=True)


    # Check that the primary threshold is one of the simulated ones
    if not np.any(np.isclose(PVALS, args.pthr)):
        raise ValueError(f"--pthr {args.pthr} is not one of the simulated p-values {PVALS}.")

    # Check if the number of iterations is positive
    if args.n_iterations <= 0:
        raise ValueError(f"Number of iterations {args.n_iterations} is not positive.")
    # Check if the number of iterations is an integer
    if not isinstance(args.n_iterations, int):
        raise TypeError(f"Number of iterations {args.n_iterations} is not an integer.")

    return args


def _check_afni():
    # Check if the env var $AFNI_IMAGE_PATH is set
    if "AFNI_IMAGE_PATH" not in os.environ:
        raise EnvironmentError(
        "$AFNI_IMAGE_PATH is not set. Please set it to the AFNI apptainer image path."
        )

    # Check if the env var $AFNI_IMAGE_PATH is a file that exists
    if not os.path.isfile(os.environ["AFNI_IMAGE_PATH"]):
        raise FileNotFoundError(
            f"$AFNI_IMAGE_PATH {os.environ['AFNI_IMAGE_PATH']} does not exist."
        )

    afni_path = os.environ["AFNI_IMAGE_PATH"]

    return afni_path


def _check_fsl():
    # Check if the env var $FSLDIR is set
    if "FSLDIR" not in os.environ:
        raise EnvironmentError(
            "$FSLDIR is not set. Please set it to the FSL installation path."
        )

    # Check if the env var $FSLDIR is a directory that exists
    if not os.path.isdir(os.environ["FSLDIR"]):
        raise FileNotFoundError(
            f"$FSLDIR {os.environ['FSLDIR']} does not exist."
        )

    fsl_bin = os.path.join(os.environ["FSLDIR"], "bin")

    return fsl_bin


# Many of the commands in this script were kindly provided by @ajimenezmarin, thanks Antonio c:
def main():
    parser = _setup_parser()
    args = _check_args(parser)
    afni_path = _check_afni()
    fsl_bin = _check_fsl()

    # Estimate the smoothness of GLM residuals
    print("\n[INFO] Estimating smoothness of GLM residuals...")
    # Construct the bind mounts to any external paths
    binds = ""
    for i, arg in enumerate([args.mask, args.residuals, args.output, args.zmap]):
        binds += f" --bind {os.path.dirname(os.path.abspath(arg))}"
    binds = binds[1:]

    command = (
        f"cd {args.output} && "
        f"apptainer exec {binds} {afni_path} 3dFWHMx -mask {args.mask} -acf tmp.txt {args.residuals} | "
        "tail -n 1 | awk '{print $1\" \"$2\" \"$3}' > acf.txt"
    )
    if os.path.exists(os.path.join(args.output, "acf.txt")) and not args.recompute:
        print("[INFO] acf.txt already exists. Skipping smoothness estimation.")
    else:
        sp.run(command, shell=True, check=True)

    # Run 3dClustSim to estimate the cluster size threshold
    print(("\n[INFO] Running 3dClustSim to estimate the cluster size threshold for"
           "different significance levels..."))
    pvals_string = " ".join([str(p) for p in PVALS])

    command = (
        f"cd {args.output} && "
        f"apptainer exec {binds} {afni_path} 3dClustSim -mask {args.mask} -acf $(cat acf.txt) -athr {PVALS[0]} "
        f"-iter {args.n_iterations} -pthr {pvals_string} -prefix acf"
    )

    acf_nn2_2sided_1d = os.path.join(args.output, "acf.NN2_2sided.1D")
    if os.path.exists(acf_nn2_2sided_1d) and not args.recompute:
        print("[INFO] acf.NN2_2sided.1D already exists. Skipping 3dClustSim.")
    else:
        sp.run(command, shell=True, check=True)

    # Cleanup
    sp.run(f"cd {args.output} && rm tmp.*", shell=True, check=False)

    # Read the NN2_2sided.1D cluster size table
    print("\n[INFO] Reading the NN2_2sided.1D cluster size table...")
    acf_nn2_2sided_1d = np.loadtxt(acf_nn2_2sided_1d)
    acf_nn2_2sided_1d = pl.DataFrame(acf_nn2_2sided_1d).rename({
        "column_0": "p_value",
        "column_1": "cluster_size",
    })

    def _correct_at(pvalue, cluster_size, out_name):
        """Keep the clusters of |Z| > z(p/2) (NN2 = 18-connectivity) with >= k voxels."""
        z_thresh = float(norm.ppf(1 - (pvalue / 2)))
        print((f"[INFO] Filtering clusters at Z-threshold: {z_thresh:.4f}"
               f" (two-sided p < {pvalue}), removing clusters below {cluster_size} voxels"))
        command = (
            f"cd {args.output} && "
            # Detect the clusters. --connectivity=18 matches the NN2 (faces + edges) table of
            # 3dClustSim; FSL's default (26) would merge corner-touching clusters.
            f"{fsl_bin}/fsl-cluster --in={args.zmap} --thresh={z_thresh} --connectivity=18 "
            f"--osize=Csize_{pvalue} --no_table && "
            # Keep clusters with size >= k and binarise
            f"{fsl_bin}/fslmaths Csize_{pvalue} -thr {cluster_size} -bin {out_name}"
        )
        sp.run(command, shell=True, check=True)

    # Find the cluster-size threshold for the primary p-threshold
    table = {float(p): float(k) for p, k in acf_nn2_2sided_1d.rows()}
    pthr = next(p for p in table if np.isclose(p, args.pthr))
    k_thr = table[pthr]

    # Corrected map at the single, a-priori primary threshold.
    # NOTE: this used to be the union of the surviving clusters over all 11 thresholds,
    # which does not control FWE at 0.05 (each threshold does so only on its own).
    print("\n[INFO] Correcting the clusters at the primary threshold...")
    _correct_at(pthr, k_thr, "NN2_2sided_results_mask.nii.gz")

    with open(os.path.join(args.output, "cluster_correction_info.txt"), "w") as f:
        f.write(f"zmap\t{args.zmap}\n")
        f.write(f"pthr_two_sided\t{pthr}\n")
        f.write(f"z_thresh\t{float(norm.ppf(1 - pthr / 2)):.6f}\n")
        f.write(f"cluster_size_k\t{k_thr}\n")
        f.write("connectivity\tNN2 (18)\n")
        f.write("alpha_cluster\t0.05\n")

    if args.save_all_pthr:
        print("\n[INFO] Saving exploratory per-threshold maps (NOT combined)...")
        for pvalue, cluster_size in table.items():
            _correct_at(pvalue, cluster_size, f"acf_NN2_2sided_pval_{pvalue}.nii.gz")

    # Cleanup
    lookup_pattern = os.path.abspath(os.path.join(args.output, "acf.NN*"))
    acf_NN_files = [to_remove for to_remove in glob(lookup_pattern) if "NN2_2sided" not in to_remove]
    for file in acf_NN_files:
        os.remove(file)

    print(("\n[INFO] DONE! See the output directory for the results:\n"
           f"{args.output}"))


if __name__ == "__main__":
    main()