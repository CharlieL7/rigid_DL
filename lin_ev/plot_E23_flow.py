import argparse as argp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def main():
    parser = argp.ArgumentParser(description="E23 flow plot in 2D with ellipsoid insert")
    parser.add_argument("dims", nargs=3, type=float, help="ellipsoid dimensions")
    parser.add_argument(
        "-o",
        "--out_tag",
        help="tag to prepend to output plot",
        default="E23_flow_2D_quiver"
    )
    args = parser.parse_args()


    E = E_23(args.dims)
    E = E[1:3, 1:3]
    
    fig = plt.figure(figsize=[3, 3])
    ax = fig.add_subplot(111)

    y, z = np.meshgrid(
        np.arange(-4., 4.1, 1.0),
        np.arange(-4., 4.1, 1.0)
    )
    pos = np.array([y, z])

    vels = np.einsum("ij, jlm -> ilm", E, pos)
    print(vels)
    ax.quiver(y, z, vels[0], vels[1])

    ellipse = Ellipse([0., 0.,], args.dims[1], args.dims[2])

    ax.add_artist(ellipse)
    ellipse.set_alpha(0.70)

    add_textbox(ax, "{}-{}-{}".format(int(args.dims[0]), int(args.dims[1]), int(args.dims[2])))

    ax.set_title(r"$E^{(23)}$")
    ax.set_xlabel(r"$y$")
    ax.set_ylabel(r"$z$")
    fig.tight_layout(rect=[0, 0, 1.00, 1])
    fig.savefig("{}.pdf".format(args.out_tag), format="pdf")


def E_23(dims):
    """
    Off diagonal (23) rate of strain field eigenfunction

    Parameters:
        dims: ellipsoid dimensions
    Returns
    """
    E_d = np.array([[0., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
    _a, b, c = dims
    E_add = np.array([[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
    E = E_d + ((b**2 - c**2)/(b**2 + c**2)) * E_add
    print(E)
    return E


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
    ax.text(0.90, 1.10, textstr,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=10,
            bbox=props
            )


if __name__ == "__main__":
    main()
