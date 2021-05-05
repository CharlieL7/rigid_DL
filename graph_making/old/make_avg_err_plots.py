import glob
import argparse as argp
import numpy as np
import meshio
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

NUM_ELE = [80, 180, 320, 500, 720]

def main():
    parser = argp.ArgumentParser(description="Creates boxplots from a directory of vtk files with local absolute error data")
    parser.add_argument("in_dir", help="folder of vtk files")
    parser.add_argument(
        "-o",
        "--outname",
        help="output file name",
        default="avg_rel_err"
    )
    parser.add_argument(
        "-t",
        "--tag",
        help="tag to put on figure",
        required=True
    )
    args = parser.parse_args()
    folder = args.in_dir
    E12_errs = []
    E23_errs = []
    E31_errs = []
    Ep_errs = []
    Em_errs = []
    H3x3_0_errs = []
    H3x3_1_errs = []
    H3x3_2_errs = []
    for vtk_file in sorted(glob.glob(folder + "*.vtk")):
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

    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    ax.plot(NUM_ELE, E12_errs, label=r"$E^{(12)}$", marker=".")
    ax.plot(NUM_ELE, E23_errs, label=r"$E^{(23)}$", marker=".")
    ax.plot(NUM_ELE, E31_errs, label=r"$E^{(31)}$", marker=".")
    ax.plot(NUM_ELE, Ep_errs, label=r"$E^{(+)}$", marker=".")
    ax.plot(NUM_ELE, Em_errs, label=r"$E^{(-)}$", marker=".")
    ax.plot(NUM_ELE, H3x3_0_errs, label=r"$Q^{(1)}$", marker=".")
    ax.plot(NUM_ELE, H3x3_1_errs, label=r"$Q^{(2)}$", marker=".")
    ax.plot(NUM_ELE, H3x3_2_errs, label=r"$Q^{(3)}$", marker=".")
    ax.set_ylim([0., 0.2])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.))
    ax.set_xlabel("Number of elements")
    ax.set_ylabel("Average relative local error")
    ax.legend(loc="upper right")
    add_textbox(ax, args.tag)
    fig.tight_layout(rect=[0, 0, 0.95, 1])
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
    ax.text(0.25, 0.90, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)


def c2l(x):
    if isinstance(x, list):
        return x[0]
    return x


if __name__ == "__main__":
    main()
