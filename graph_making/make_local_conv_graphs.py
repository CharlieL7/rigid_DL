import pandas as pd
import argparse as argp
import matplotlib.pyplot as plt

def main():
    parser = argp.ArgumentParser(description="Creates 1x2 plot of average and maximum local error data")
    parser.add_argument("avg_csv", help="Average local error data csv file")
    parser.add_argument("max_csv", help="Maximum local error data csv file")
    parser.add_argument(
        "-o",
        "--outname",
        help="output pdf file name",
        default="local_errors"
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
    print(csv_files)
    for i, fn in enumerate(csv_files):
        df = pd.read_csv(fn)
        print(df)
                
