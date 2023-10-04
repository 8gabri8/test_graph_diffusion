#!/usr/bin/env python3

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from nigsp import io, viz
from nigsp.operations import laplacian
from nigsp.operations.timeseries import resize_ts

FIGSIZE = (50, 25)


# Needs nigsp >= 0.17
def _get_parser():
    """
    Parse command line inputs for this function.

    Returns
    -------
    parser.parse_args() : argparse dict

    """
    parser = argparse.ArgumentParser(
        description=(
            "Runs scalar tau estimation with any number of given matrices, either with Ordinary or Generalised Least Squares."
        ),
        add_help=False,
    )
    required = parser.add_argument_group("Required Arguments")
    required.add_argument(
        "-f",
        "--input-func",
        dest="tsfile",
        type=str,
        help=(
            "Complete path (absolute or relative) and name "
            "of the file containing fMRI signal."
        ),
        required=True,
    )
    required.add_argument(
        "-s",
        "--input-structural",
        dest="structural_files",
        action="extend",
        nargs="+",
        type=str,
        help=(
            "Complete path (absolute or relative) and name "
            "of the file containing the structural "
            "connectivity matrices. This file "
            "can be a 1D, txt, csv, tsv, or mat file."
        ),
        required=True,
    )

    optional = parser.add_argument_group("Optional Arguments")

    optional.add_argument(
        "-not0",
        "--no-tau-0",
        dest="add_tau0",
        action="store_false",
        help=("Do not add to the model tau0, associated to I @ ts_{n-1}."),
        default=True,
    )
    optional.add_argument(
        "-od",
        "--outdir",
        dest="odr",
        type=str,
        help=(
            "Output folder. If None, is specified, a folder `crispy_scalar` will be "
            "created in the folder of the timeseries file."
        ),
        default=None,
    )
    optional.add_argument(
        "-sub",
        "--subject",
        dest="sub",
        type=str,
        help=("Subject name"),
        default="random",
    )
    optional.add_argument(
        "-gls",
        "--general-least-square",
        dest="gls",
        action="store_true",
        help=(
            "After a first pass of OLS, add the covariate matrix of the innovation "
            "signal to improve tau estimation. This will start a recursive estimation, "
            "until either tolerance level or the maximum amount of runs are reached."
        ),
        default=False,
    )
    optional.add_argument(
        "-tol",
        "--tolerance",
        dest="tol",
        type=float,
        help=("Tolerance level. Default 0."),
        default=0,
    )
    optional.add_argument(
        "-max",
        "--max-run",
        dest="max_run",
        type=int,
        help=("Maximum number of reiterations. Default 5000."),
        default=5000,
    )

    optional.add_argument(
        "-h", "--help", action="help", help="Show this help message and exit"
    )
    return parser


# Define some functions to simplify later math
def tr(x, y):
    """Compute Tr(x^T y)."""
    return np.trace(x.T @ y)


def plot_compl(y, norm_time, norm_space, title, filename):
    gs_kw = dict(width_ratios=[0.9, 0.1], height_ratios=[0.2, 0.8])
    yax = np.arange(len(norm_space))
    f, ax = plt.subplots(nrows=2, ncols=2, gridspec_kw=gs_kw, figsize=FIGSIZE)
    ax[1, 0].imshow(y, cmap="gray")
    ax[0, 0].plot(norm_time)
    ax[1, 1].plot(norm_space, yax)
    ax[0, 0].set_xlim(0, len(norm_time) - 1)
    ax[1, 1].set_ylim(0, yax[-1])
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return


def numerical_estimation_scalar_tau(ts_orig, phi, covmat=None):
    keys = list(phi.keys())

    if covmat is None:
        covmat = np.diag(np.ones(phi[1].shape[0]))
    else:
        covmat = np.linalg.pinv(covmat)

    C = np.empty((len(keys), len(keys)))
    D = np.empty((len(keys)))

    print("Compute tau_s_ coefficients")
    ts_diff = np.diff(ts_orig)
    for n, j in enumerate(keys):
        D[n] = -tr(ts_diff, covmat @ phi[j])
        for m, k in enumerate(keys):
            C[m, n] = tr(phi[k], covmat @ phi[j])

    print("Estimate tau_s_ through numerical model")
    tau = np.linalg.pinv(C) @ D

    return tau


def innov_estimation_scalar_tau(ts_orig, phi, tau):
    keys = list(phi.keys())

    print("Reconstruct innovation timeseries")
    ts_innov = np.diff(ts_orig)
    for n, k in enumerate(keys):
        ts_innov = ts_innov + tau[n] * phi[k]

    return ts_innov


def multitau_gls_estimation(
    tsfile,
    structural_files,
    add_tau0=True,
    odr=None,
    sub="random",
    gls=False,
    tol=0,
    max_run=5000,
):
    # ## Check the required variables
    if not tsfile:
        raise ValueError("Missing timeseries input.")
    structural_files = (
        structural_files if type(structural_files) is list else [structural_files]
    )
    if len(structural_files) < 1:
        raise ValueError("Not enough tau-related matrices declared.")

    # ## Read data, transform into Laplacian, and decompose.
    keys = list(range(1, len(structural_files) + 1))

    d = dict.fromkeys(keys)
    lapl = dict.fromkeys(keys)
    lapl_norm = dict.fromkeys(keys)

    # Check inputs type
    print("Check input data")
    sc_is = dict.fromkeys(io.EXT_DICT.keys(), "")
    ts_is = dict.fromkeys(io.EXT_DICT.keys(), False)
    for k in io.EXT_DICT.keys():
        ts_is[k], _ = io.check_ext(io.EXT_DICT[k], tsfile)
        sc_is[k] = []
        for f in structural_files:
            sc_is[k] += [io.check_ext(io.EXT_DICT[k], f)[0]]
        sc_is[k] = all(sc_is[k])

    # Prepare structural connectivity matrix input and read in functional data
    loadfunc = None
    ts = None
    for ftype in io.LOADMAT_DICT.keys():
        if sc_is[ftype]:
            print(f"Structure files will be loaded as an {ftype} file")
            loadfunc = io.LOADMAT_DICT[ftype]
        if ts_is[ftype]:
            ts = {}
            print(f"Load {os.path.basename(tsfile)} as an {ftype} file")
            ts["orig"] = io.LOADMAT_DICT[ftype](tsfile)

    if loadfunc is None:
        raise NotImplementedError("Input structural file is not in a supported type.")

    if ts is None:
        raise NotImplementedError(f"Input file {tsfile} is not of a supported type.")

    # Create folder before it's too late
    # If odr is None, save in the folder of the ts file.
    if odr is None:
        odr = os.path.join(os.path.dirname(tsfile), "crispy_scalar")

    os.makedirs(f"{odr}/plots", exist_ok=True)
    os.makedirs(f"{odr}/files", exist_ok=True)

    # Column-center ts (sort of GSR)
    print("Column center timeseries (GSR)")
    ts["orig"] = resize_ts(ts["orig"], resize="norm")
    ts["orig"] = ts["orig"] - ts["orig"].mean(axis=0)[np.newaxis, ...]

    # ### Create SC matrix
    print("Laplacian-ise the structural files")
    for n, k in enumerate(keys):
        d[k] = loadfunc(structural_files[n], shape="square")
        lapl[k], degree = laplacian.compute_laplacian(d[k], selfloops="degree")
        lapl_norm[k] = laplacian.normalisation(lapl[k], degree, norm="rwo")

    # ## Compute phi to simplify coefficients
    phi = dict.fromkeys(keys)

    for k in keys:
        phi[k] = lapl_norm[k] @ ts["orig"][:, :-1]

    if add_tau0:
        print("Add t0 to the model")
        keys = [0] + keys
        lapl_norm[0] = np.diag(np.ones(lapl_norm[1].shape[0]))
        phi[0] = ts["orig"][:, :-1].copy()

    tau = numerical_estimation_scalar_tau(ts["orig"], phi)
    ts["innov"] = innov_estimation_scalar_tau(ts["orig"], phi, tau)

    if gls:
        covmat = {"ols": np.cov(ts["innov"].copy())}
        for n in range(max_run):
            print("Start GLS estimation using previous covariance")
            cov = np.cov(ts["innov"])
            ts_prev = ts["innov"].copy()

            tau = numerical_estimation_scalar_tau(ts["orig"], phi, cov)
            ts["innov"] = innov_estimation_scalar_tau(ts["orig"], phi, tau)

            t = np.linalg.norm(ts_prev - ts["innov"])

            print(f"Round {n}, tolerance: {t}")
            if t <= tol:
                break
        covmat["gls"] = np.cov(ts["innov"]).copy()
        io.export_txt(np.asarray([n, t]), f"{odr}/sub-{sub}_gls_estimation_log.txt")

    print("Compute norm")
    norm = {}
    for k in ts.keys():
        norm[f"{k}_time"] = np.linalg.norm(ts[k], axis=0)
        norm[f"{k}_space"] = np.linalg.norm(ts[k], axis=-1)

    # ## Plot and export everything

    io.export_txt(tau, f"{odr}/files/sub-{sub}_tau_scalar.tsv")

    # If a previous run created a lapl_norm_0, delete it.
    if not add_tau0 and os.path.exists(f"{odr}/files/sub-{sub}_lapl_norm_0.tsv.gz"):
        print(
            f"Remove existing tau0 file from a previous run: {odr}/files/sub-{sub}_lapl_norm_0.tsv.gz"
        )
        os.remove(f"{odr}/files/sub-{sub}_lapl_norm_0.tsv.gz")

    print("Save everything")
    for k in ts.keys():
        io.export_txt(ts[k], f"{odr}/files/sub-{sub}_ts-{k}.tsv.gz")
        for nt in ["time", "space"]:
            io.export_txt(
                norm[f"{k}_{nt}"], f"{odr}/files/sub-{sub}_ts-{k}_norm_{nt}.1D"
            )

        plot_compl(
            ts[k],
            norm[f"{k}_time"],
            norm[f"{k}_space"],
            title=f'Timeseries "{k}" sub {sub}, estimated taus: {tau}',
            filename=f"{odr}/plots/sub-{sub}_ts-{k}.png",
        )

    for k in keys:
        io.export_txt(lapl_norm[k], f"{odr}/files/sub-{sub}_lapl_norm_{k}.tsv.gz")

    if gls:
        for k in covmat.keys():
            io.export_txt(
                covmat[k], f"{odr}/files/sub-{sub}_ts-innov_covmat-{k}.tsv.gz"
            )
            viz.plot_connectivity(
                covmat[k], f"{odr}/plots/sub-{sub}_ts-innov_covmat-{k}.png"
            )


def _main(argv=None):
    options = _get_parser().parse_args(argv)

    multitau_gls_estimation(**vars(options))


if __name__ == "__main__":
    _main(sys.argv[1:])
