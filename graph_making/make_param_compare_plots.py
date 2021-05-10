import argparse as argp
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argp.ArgumentParser(description="Creates 1x2 plot of average and maximum local error data")
    parser.add_argument("avg_csv", help="Average local error data csv file")
    parser.add_argument("max_csv", help="Maximum local error data csv file")
    parser.add_argument(
        "-o",
        "--outname",
        help="output pdf file name",
        default="param_errors"
    )
    parser.add_argument(
        "-t",
        "--tag",
        help="Ellipsoid dimensions tag to put on figure",
        required=True
    )
    args = parser.parse_args()

    fig, axs = plt.subplots(1, 2, figsize=(6, 4.5))

    csv_files = [args.avg_csv, args.max_csv]
    for i, fn in enumerate(csv_files):
        df = pd.read_csv(fn)
        num_nodes = df["number of nodes"]
        axs[i].plot(num_nodes, df["cp-le"], label="cp-le")
        axs[i].plot(num_nodes, df["lp-le"], label="lp-le")
        axs[i].plot(num_nodes, df["cp-qe"], label="cp-qe")
        axs[i].plot(num_nodes, df["lp-qe"], label="lp-qe")

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    fig.text(0.965, 0.90, args.tag,
        horizontalalignment="center",
        verticalalignment="center",
        bbox=props
    )
    fig.tight_layout(rect=[0, 0, 0.95, 1])
    fig.savefig("{}.pdf".format(args.outname), format="pdf")

if __name__ == "__main__":
    main()
