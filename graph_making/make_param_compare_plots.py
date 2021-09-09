import argparse as argp
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.text as text
import matplotlib.lines as lines

param_type_dict = {
    "cp-le": "o",
    "lp-le": "^",
    "cp-qe": "s",
    "lp-qe": "p",
}

flow_key_arr = [
    "E12",
    "E23",
    "E31",
    "Ep",
    "Em",
    "3x3_0",
    "3x3_1",
    "3x3_2",
]
flow_strings = [
    r"$E^{(12)}$",
    r"$E^{(23)}$",
    r"$E^{(31)}$",
    r"$E^{(+)}$",
    r"$E^{(-)}$",
    r"$Q^{(1)}$",
    r"$Q^{(2)}$",
    r"$Q^{(3)}$",
]

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

def main():
    parser = argp.ArgumentParser(description="Creates 1x2 plots of average and maximum local error data")
    parser.add_argument("err_json", help="Local error data json file generated from extract_local_error_data.py")
    parser.add_argument(
        "-d",
        "--dims",
        help="Ellipsoid dimensions as a tag (ex. 4-1-1)",
        required=True
    )
    args = parser.parse_args()

    with open(args.err_json, "r") as read_file:
        data = json.load(read_file)

    for i, flow_key in enumerate(flow_key_arr):
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        axs[0].set_xscale("log")
        axs[0].set_yscale("log")

        axs[0].set_xlabel("Number of nodes")
        axs[0].set_ylabel("Mean local error")

        axs[1].set_xlabel("Number of nodes")
        axs[1].set_ylabel("Max local error")

        for param_type, mark_type in param_type_dict.items():
            line_data = data[param_type][flow_key]
            axs[0].plot(line_data["num_nodes"], line_data["avg_loc_err"], label=param_type, marker=mark_type, markersize=3)
            axs[1].plot(line_data["num_nodes"], line_data["max_loc_err"], label=param_type, marker=mark_type, markersize=3)

        fig.text(0.85, 0.82,
            flow_strings[i] + " " +  args.dims,
            horizontalalignment="center",
            verticalalignment="center",
            bbox=props
        )
        add_slope_tri(axs[0])
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.85, 0.62))
        fig.tight_layout(rect=[0, 0, 1.00, 1])
        fig.savefig("{}_{}_errors.pdf".format(args.dims, flow_key), format="pdf")


def add_slope_tri(ax):
    line = lines.Line2D(np.array([.30, .30, .40]), np.array([0.40, 0.30, 0.30]),
            lw = 1, color="black", transform=ax.transAxes)
    ax.add_line(line)
    t = text.Text(0.35, 0.26, "-1", axes=ax, ha="center", va="center", transform=ax.transAxes, size=8)
    ax.add_artist(t)


if __name__ == "__main__":
    main()
