import glob
import os
import argparse as argp
import numpy as np
import meshio
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

PARAM_LIST = ["4_1_1_cp_le_s3_E12", "4_1_1_lp_le_s3_E12"]

def main():
    parser = argp.ArgumentParser(description="Makes histogram from two vtk data files")
    parser.add_argument("file_0", help="first vtk file")
    parser.add_argument("file_1", help="second vtk file")
    parser.add_argument(
        "-o",
        "--outname",
        help="output file name",
        default="hist_rel_errs"
    )
    args = parser.parse_args()

    file_names = [args.file_0, args.file_1]
    fig, axs = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(6, 6))
    axs = np.ravel(axs)

    E12_errs = []
    for i, fn in enumerate(file_names):
        io_mesh = meshio.read(fn)
        if io_mesh.cell_data:
            data = io_mesh.cell_data
        elif io_mesh.point_data:
            data = io_mesh.point_data
        E12_errs.append(data["rel_err_E12"])
    max_error = np.max([np.max(E12_errs[0]), np.max(E12_errs[1])])

    for i in range(len(E12_errs)):
        bins_range = np.linspace(0, max_error, 20)
        axs[i].hist(E12_errs[i], bins=bins_range)

        axs[i].set_xlabel("Relative local error")

        axs[i].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.))
        axs[i].set_title(PARAM_LIST[i])
    axs[0].set_ylabel("Number of nodes")

    fig.tight_layout(rect=[0, 0, 0.95, 1])
    fig.savefig("{}_hist_E12.pdf".format(args.outname), format="pdf")


def add_textbox(ax, textstr):
    """
    Adds an informative textbox to to figure

    Parameters:
        ax: the axes object
        textstr: the string in the textbox
    Returns:
        None
    """
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.25, 0.90, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)


def c2l(x):
    if isinstance(x, list):
        return x[0]
    return x

if __name__ == "__main__":
    main()
