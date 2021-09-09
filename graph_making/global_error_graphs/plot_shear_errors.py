import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

num_ele = [80, 180, 320, 500, 720]
cp_le_errs = [4.7896E-02, 2.0524E-02, 1.2096E-02, 6.9916E-03, 4.6580E-03, 3.2833E-03, 2.4462E-03, 1.8917E-03, 1.5073E-03]
lp_le_errs = [3.5350E-01, 5.3473E-02, 2.1316E-02, 7.4484E-03, 3.7095E-03, 2.0009E-03, 1.2057E-03, 7.7830E-04, 5.3293E-04]
cp_qe_errs = [3.2325E-02, 1.4569E-02, 6.7584E-03, 4.4295E-03, 2.8950E-03, 2.0547E-03, 1.5179E-03, 1.1665E-03, 9.2319E-04]
lp_qe_errs = [1.2110E-02, 8.1532E-05, 6.3375E-04, 5.1158E-04, 4.3231E-05, 1.3293E-04, 2.2505E-04, 2.5636E-04, 2.6223E-04]
cp_le_nodes = [80, 180, 320, 500, 720, 980, 1280, 1620, 2000]
lp_le_nodes = [42, 92, 162, 252, 362, 492, 642, 812, 1002]
cp_qe_nodes = [80, 180, 320, 500, 720, 980, 1280, 1620, 2000]
lp_qe_nodes = [42, 92, 162, 252, 362, 492, 642, 812, 1002]


def main():
    #fig = plt.figure(figsize=(3.5, 3.5))
    #ax = fig.add_subplot(111)
    #ax.plot(num_ele, cp_le_errs, label="cp-le", marker=".")
    #ax.plot(num_ele, lp_le_errs, label="lp-le", marker=".")
    #ax.plot(num_ele, cp_qe_errs, label="cp-qe", marker=".")
    #ax.plot(num_ele, lp_qe_errs, label="lp-qe", marker=".")
    ##abline(-1)
    #ax.set_xlabel("Number of elements")
    #ax.set_ylabel("Relative rotational velocity error")
    #ax.legend(loc="upper right")
    #fig.tight_layout(rect=[0, 0, 0.95, 1])

    #ax.set_xscale("log")
    #ax.set_yscale("log")
    #fig.savefig("global_shear_flow_error_loglog_scale.pdf", format="pdf")

    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    ax.plot(cp_le_nodes, cp_le_errs, label="cd-le", marker=".")
    ax.plot(lp_le_nodes, lp_le_errs, label="ld-le", marker=".")
    ax.plot(cp_qe_nodes, cp_qe_errs, label="cp-qe", marker=".")
    ax.plot(lp_qe_nodes, lp_qe_errs, label="lp-qe", marker=".")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.))
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Relative rotational velocity error")
    ax.legend(loc="upper right")
    fig.tight_layout(rect=[0, 0, 0.95, 1])
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.savefig("global_shear_flow_error_loglog_nodes.pdf", format="pdf")


if __name__ == "__main__":
    main()
