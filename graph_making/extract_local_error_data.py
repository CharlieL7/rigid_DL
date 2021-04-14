"""
Extracts local error data in the form of average and maximum local error
for a directory of vtk files. Intended to output a csv file for a given
parameterization to the columns: number of elements, average local error, maximum
local error. Appends a section to a csv file for each flow type.
"""

import csv
import os
import argparse as argp
import numpy as np
import glob
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
        "-o",
        "-out_name",
        help="Output csv file name",
        default="local_err_data"
    )
    args = parser.parse_args()
    data_dict = {
        "E12": [],
        "E23": [],
        "E31": [],
        "Ep": [],
        "Em": [],
        "3x3_0": [],
        "3x3_1": [],
        "3x3_2": [],
    }
    
    in_dir = os.path.join(args.in_dir, "")
    assert os.path.isdir(in_dir)
    for vtk_file in sorted(glob.glob(in_dir + "*.vtk")):
        io_mesh = meshio.read(vtk_file)
        assert len(io_mesh.cells) == 1
        num_ele = len(io_mesh.cells[0].data)
        if io_mesh.cell_data:
            data = io_mesh.cell_data
        elif io_mesh.point_data:
            data = io_mesh.point_data
        for flow_key in data_dict:
            data_dict[flow_key].append([
                num_ele,
                np.mean(data["local_err_" + flow_key]),
                np.max(data["local_err_" + flow_key]),
            ])

    field_names = [
        "number of elements",
        "average local error",
        "max local error",
    ]
    with open("{}.csv".format(args.o), "w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=",", lineterminator=os.linesep)
        for flow_key in data_dict:
            csv_file.write("Flow type: " + flow_key + "\n")
            writer.writerow(field_names)
            writer.writerows(data_dict[flow_key])
            csv_file.write("\n")


if __name__ == "__main__":
    main()
