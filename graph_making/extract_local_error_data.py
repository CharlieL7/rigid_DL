"""
Extracts local error data in the form of average and maximum local error
for a directory of vtk files. Intended to output a json file for a given
parameterization.
"""
import json
import os
import sys
import copy
import argparse as argp
import glob
import numpy as np
import meshio

def main():
    parser = argp.ArgumentParser(
        description="Reads a directory of vtk files of a given " +
        "parameterization to output a csv file to make convergence " +
        "graphs."
    )
    parser.add_argument(
        "in_dir",
        help="Directory of vtk files to read"
    )
    parser.add_argument(
        "param_type",
        help="Parameterization type of the potentials"
    )
    parser.add_argument(
        "-o",
        "-out_name",
        help="Output csv file name",
        default="local_err_data"
    )
    args = parser.parse_args()

    if args.param_type in ["cp-le", "cp-qe"]:
        pot_type = 0
    elif args.param_type in ["lp-le", "lp-qe"]:
        pot_type = 1
    elif args.param_type in ["qp-le", "qp-qe"]:
        pot_type = 2
    else:
        sys.exit("Unknown parameterization type: {}".format(args.param_type))

    struct_dict = {
        "num_nodes": [],
        "num_ele": [],
        "avg_loc_err": [],
        "max_loc_err": [],
    }
    data_dict = {
        args.param_type: {
            "E12": copy.deepcopy(struct_dict),
            "E23": copy.deepcopy(struct_dict),
            "E31": copy.deepcopy(struct_dict),
            "Ep": copy.deepcopy(struct_dict),
            "Em": copy.deepcopy(struct_dict),
            "3x3_0": copy.deepcopy(struct_dict),
            "3x3_1": copy.deepcopy(struct_dict),
            "3x3_2": copy.deepcopy(struct_dict),
        }
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

    in_dir = os.path.join(args.in_dir, "")
    assert os.path.isdir(in_dir)
    for vtk_file in sorted(glob.glob(in_dir + "*.vtk")):
        io_mesh = meshio.read(vtk_file)
        assert len(io_mesh.cells) == 1
        _num_ele = len(io_mesh.cells[0].data)
        if pot_type == 0:
            _num_nodes = _num_ele
        elif pot_type == 1:
            _num_nodes = _num_ele / 2 + 2
        elif pot_type == 2:
            _num_nodes = _num_ele * 2 + 2

        if io_mesh.cell_data:
            data = io_mesh.cell_data
        elif io_mesh.point_data:
            data = io_mesh.point_data
        for flow_key in flow_key_arr:
            data_dict[args.param_type][flow_key]["num_nodes"].append(_num_nodes)
            data_dict[args.param_type][flow_key]["num_ele"].append(_num_ele)
            data_dict[args.param_type][flow_key]["avg_loc_err"].append(np.mean(data["local_err_" + flow_key]))
            data_dict[args.param_type][flow_key]["max_loc_err"].append(np.max(data["local_err_" + flow_key]))

    with open("{}.json".format(args.o), "w", newline="") as json_file:
        json.dump(data_dict, json_file)


if __name__ == "__main__":
    main()
