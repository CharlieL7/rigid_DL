import glob
import os
import argparse as argp
import numpy as np
import meshio
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import scipy.stats

NUM_ELE = [80, 180, 320, 500, 720]
FOLDER_NAMES = ["cp_le", "cp_qe", "lp_le", "lp_qe"]

def main():
    parser = argp.ArgumentParser(description="Creates 2x2 plot of average local error data for the 4 parameterizations")
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

    fig_line, axs_line = plt.subplots(2, 2, sharey=True, figsize=(6, 6))
    fig_quad, axs_quad = plt.subplots(2, 2, sharey=True, figsize=(6, 6))
    axs_line = np.ravel(axs_line)
    axs_quad = np.ravel(axs_quad)

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
            E12_errs.append(np.mean(data["rel_err_E12"]))
            E23_errs.append(np.mean(data["rel_err_E23"]))
            E31_errs.append(np.mean(data["rel_err_E31"]))
            Ep_errs.append(np.mean(data["rel_err_Ep"]))
            Em_errs.append(np.mean(data["rel_err_Em"]))
            H3x3_0_errs.append(np.mean(data["rel_err_3x3_0"]))
            H3x3_1_errs.append(np.mean(data["rel_err_3x3_1"]))
            H3x3_2_errs.append(np.mean(data["rel_err_3x3_2"]))

        axs_line[i].plot(NUM_ELE, E12_errs, label=r"$E^{(12)}$", marker=".")
        axs_line[i].plot(NUM_ELE, E23_errs, label=r"$E^{(23)}$", marker=".")
        axs_line[i].plot(NUM_ELE, E31_errs, label=r"$E^{(31)}$", marker=".")
        axs_line[i].plot(NUM_ELE, Ep_errs, label=r"$E^{(+)}$", marker=".")
        axs_line[i].plot(NUM_ELE, Em_errs, label=r"$E^{(-)}$", marker=".")

        axs_quad[i].plot(NUM_ELE, H3x3_0_errs, label=r"$Q^{(1)}$", marker=".")
        axs_quad[i].plot(NUM_ELE, H3x3_1_errs, label=r"$Q^{(2)}$", marker=".")
        axs_quad[i].plot(NUM_ELE, H3x3_2_errs, label=r"$Q^{(3)}$", marker=".")

        axs_line[i].set_xscale("log")
        axs_line[i].set_yscale("log")
        axs_quad[i].set_xscale("log")
        axs_quad[i].set_yscale("log")
        
        print(fn)
        print(scipy.stats.linregress(np.log(NUM_ELE), np.log(E12_errs))[0])

        if i in [2, 3]:
            axs_line[i].set_xlabel("Number of elements")
            axs_quad[i].set_xlabel("Number of elements")
        if i in [0, 2]:
            axs_line[i].set_ylabel("Average relative local error")
            axs_quad[i].set_ylabel("Average relative local error")

        axs_line[i].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.))
        axs_quad[i].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.))
        add_textbox(axs_line[i], args.tag + "_" + fn)
        add_textbox(axs_quad[i], args.tag + "_" + fn)

    # Set common labels
    handles, labels = axs_line[0].get_legend_handles_labels()
    fig_line.legend(handles, labels, loc="center", bbox_to_anchor=(0.92, 0.5))

    fig_line.tight_layout(rect=[0, 0, 0.85, 1])
    fig_line.savefig("{}_line_flows_log-log.pdf".format(args.outname), format="pdf")

    handles, labels = axs_quad[0].get_legend_handles_labels()
    fig_quad.legend(handles, labels, loc="center", bbox_to_anchor=(0.92, 0.5))

    fig_quad.tight_layout(rect=[0, 0, 0.85, 1])
    fig_quad.savefig("{}_quad_flows_log-log.pdf".format(args.outname), format="pdf")

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
