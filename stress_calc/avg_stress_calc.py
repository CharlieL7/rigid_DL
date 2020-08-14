#!/usr/bin/env python3

import sys
import glob
import csv
import numpy as np
import slm_UF
import gauss_quad as gq

def main():
    if len(sys.argv) != 3:
        print("Usage: in_dir, output name")
        sys.exit()
    in_dir = sys.argv[1]
    out_name = sys.argv[2]

    Ca_x_list = []
    stress_list = []

    for dat_file in sorted(glob.glob(in_dir + "/*.dat")):
        vesicle = slm_UF.slm_UF.read_dat(dat_file)
        num_faces = vesicle.faces.shape[0]
        S = np.zeros((3, 3))
        for face_num in range(num_faces):
            x_ele = vesicle.get_nodes(vesicle.faces[face_num])
            n_ele = vesicle.calc_normal(vesicle.faces[face_num])
            v_ele = vesicle.get_vels(vesicle.faces[face_num])
            f_ele = vesicle.get_tractions(vesicle.faces[face_num])
            S += gq.int_over_tri(
                make_coeff_stresslet(x_ele, n_ele, v_ele, f_ele, vesicle.visc_rat),
                x_ele
            )
        stress_list.append(S / (vesicle.volume * vesicle.Ca))
        Ca_x_list.append(vesicle.Ca * np.sin(2 * np.pi * vesicle.De * vesicle.time))
    
    with open(out_name, "w") as out:
        writer = csv.writer(out, delimiter=",", lineterminator="\n")
        writer.writerow(["Ca_x", "S_xx", "S_xy", "S_xz", "S_yy", "S_yz", "S_zz"])
        for i in range(len(stress_list)):
            writer.writerow(
                [
                    Ca_x_list[i],
                    stress_list[i][0, 0],
                    stress_list[i][0, 1],
                    stress_list[i][0, 2],
                    stress_list[i][1, 1],
                    stress_list[i][1, 2],
                    stress_list[i][2, 2]
                ])


def make_coeff_stresslet(x_ele, n_ele, v_ele, f_ele, visc_rat):
    """
    Coefficient of the stresslet for average particle stress tensor calculation

    Parameters:
        x_ele : nodal positions (3, 3) ndarray, rows as positions
        n_ele : normal vector of linear element
        v_ele : nodal velocities (3, 3) ndarray, rows as vels
        f_ele : nodal tractions (3, 3) ndarray, rows as tractions
        visc_rat : viscosity ratio of vesicle
    Returns:
        quad_func : function to input into
        gaussian quadrature func(xi, eta)
    """

    def quad_func(xi, eta):
        f_x_0 = np.outer(f_ele[0], x_ele[0])
        f_x_1 = np.outer(f_ele[1], x_ele[1])
        f_x_2 = np.outer(f_ele[2], x_ele[2])
        A = ( # first term in coeff
            (1. - xi - eta) * f_x_0 +
            xi * f_x_1 +
            eta * f_x_2
        )
        vn_0 = np.outer(v_ele[0], n_ele)
        vn_1 = np.outer(v_ele[1], n_ele)
        vn_2 = np.outer(v_ele[2], n_ele)
        B = ( # second term in coeff
            (1. - xi - eta) * (vn_0 + np.transpose(vn_0)) +
            (xi) * (vn_1 + np.transpose(vn_1)) +
            (eta) * (vn_2 + np.transpose(vn_2))
        )
        return A - (1. - visc_rat) * B

    return quad_func


if __name__ == "__main__":
    main()
