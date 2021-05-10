import glob
import os
import argparse as argp
import numpy as np
import meshio
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import scipy.stats

NUM_ELE = [80, 180, 320, 500, 720]
FOLDER_NAMES = ["cp-le", "cp-qe", "lp-le", "lp-qe"]

def main():
    parser = argp.ArgumentParser(description="Creates plot of average local error data for the parameterizations")
    parser.add_argument("in_dir", help="Starting folder of vtk files. Expecting specific folder structure in subfolders")
    parser.add_argument(
        "-o",
        "--outname",
        help="output file name",
        default="avg_rel_errs"
    )
    parser.add_argument(
        "-t",
        "--tag",
        help="Ellipsoid dimensions tag to put on figure",
        required=True
    )
    args = parser.parse_args()

    fig, axs = plt.subplots(1, 4, sharey=True, figsize=(12, 3))
    axs = np.ravel(axs)

    for i, fn in enumerate(FOLDER_NAMES):
        subfolder = os.path.join(os.path.join(args.in_dir, fn), "")
        E12_errs = []
        E23_errs = []
        E31_errs = []
        Ep_errs = []
        Em_errs = []
        H3x3_0_errs = []
        H3x3_1_errs = []
        H3x3_2_errs = []
        for vtk_file in sorted(glob.glob(subfolder + "*.vtk")):
            io_mesh = meshio.read(vtk_file)
            if io_mesh.cell_data:
                data = io_mesh.cell_data
            elif io_mesh.point_data:
                data = io_mesh.point_data
            E12_errs.append(np.max(data["local_err_E12"]))
            E23_errs.append(np.max(data["local_err_E23"]))
            E31_errs.append(np.max(data["local_err_E31"]))
            Ep_errs.append(np.max(data["local_err_Ep"]))
            Em_errs.append(np.max(data["local_err_Em"]))
            H3x3_0_errs.append(np.max(data["local_err_3x3_0"]))
            H3x3_1_errs.append(np.max(data["local_err_3x3_1"]))
            H3x3_2_errs.append(np.max(data["local_err_3x3_2"]))

        axs[i].plot(NUM_ELE, E12_errs, label=r"$E^{(12)}$", marker="o", markersize=3)
        axs[i].plot(NUM_ELE, E23_errs, label=r"$E^{(23)}$", marker="^", markersize=3)
        axs[i].plot(NUM_ELE, E31_errs, label=r"$E^{(31)}$", marker="s", markersize=3)
        axs[i].plot(NUM_ELE, Ep_errs, label=r"$E^{(+)}$", marker="p", markersize=3)
        axs[i].plot(NUM_ELE, Em_errs, label=r"$E^{(-)}$", marker="H", markersize=3)

        axs[i].plot(NUM_ELE, H3x3_0_errs, label=r"$Q^{(1)}$", marker="o", markersize=3)
        axs[i].plot(NUM_ELE, H3x3_1_errs, label=r"$Q^{(2)}$", marker="^", markersize=3)
        axs[i].plot(NUM_ELE, H3x3_2_errs, label=r"$Q^{(3)}$", marker="s", markersize=3)

        axs[i].set_xlabel("Number of elements")
        if i in [0]:
            axs[i].set_ylabel("Maximum local error")

        add_textbox(axs[i], fn)

    # Set common labels
    props = dict(boxstyle='round', facecolor='skyblue', alpha=0.5)
    ellipsoid_tag = args.tag + "\n ellipsoid"
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.96, 0.5))
    fig.text(0.965, 0.90, ellipsoid_tag,
        horizontalalignment="center",
        verticalalignment="center",
        bbox=props
    )

    fig.tight_layout(rect=[0, 0, 0.93, 1])
    fig.savefig("{}.pdf".format(args.outname), format="pdf")


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
    ax.text(0.80, 0.90, textstr,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=10,
            bbox=props
            )


def c2l(x):
    if isinstance(x, list):
        return x[0]
    return x

if __name__ == "__main__":
    main()
