import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

num_ele = [80, 180, 320, 500, 720]
cp_le_errs = [4.7896E-02, 2.0524E-02, 1.2096E-02, 6.9916E-03, 4.6580E-03]
lp_le_errs = [3.5350E-01, 5.3473E-02, 2.1316E-02, 7.4484E-03, 3.7095E-03]
cp_qe_errs = [3.2325E-02, 1.4569E-02, 6.7584E-03, 4.4295E-03, 2.8950E-03]
lp_qe_errs = [1.2110E-02, 8.1532E-05, 6.3375E-04, 5.1158E-04, 4.3231E-05]
cp_le_nodes = [80, 180, 320, 500, 720]
lp_le_nodes = [42, 92, 162, 252, 362]
cp_qe_nodes = [80, 180, 320, 500, 720]
lp_qe_nodes = [42, 92, 162, 252, 362]


def main():
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    ax.plot(num_ele, cp_le_errs, label="cp-le", marker=".")
    ax.plot(num_ele, lp_le_errs, label="lp-le", marker=".")
    ax.plot(num_ele, cp_qe_errs, label="cp-qe", marker=".")
    ax.plot(num_ele, lp_qe_errs, label="lp-qe", marker=".")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.))
    ax.set_xlabel("Number of elements")
    ax.set_ylabel("Relative rotational velocity error")
    ax.legend(loc="upper right")
    fig.tight_layout(rect=[0, 0, 0.95, 1])
    fig.savefig("global_shear_flow_error_linear_scale.pdf", format="pdf")

    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.savefig("global_shear_flow_error_loglog_scale.pdf", format="pdf")

    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    ax.plot(cp_le_nodes, cp_le_errs, label="cp-le", marker=".")
    ax.plot(lp_le_nodes, lp_le_errs, label="lp-le", marker=".")
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
